// =============================================================================
//  Hybrid Logistics Incident Triage Engine  v2
//  fast keyword rules + optional llama.cpp fallback + human-review routing
//  + cost model + comparison-ready report
//
//  Build (rules only, no llama.cpp needed):
//      g++ -O3 -std=c++17 triage_engine.cpp -o triage
//  Build with the LLM:  make llama.h visible (auto-detected), e.g.
//      g++ -O3 -march=native -std=c++17 triage_engine.cpp -o triage
//          -I<llama.cpp>/include -I<llama.cpp>/ggml/include -L<build>/bin -lllama -lggml
//  Add -DNO_LLAMA to force rules-only.  Add -DTEST_FAKE_LLM to exercise the LLM
//  routing/report code paths WITHOUT a model (answers are fake - wiring test only).
//
//  Run:  ./triage [--csv file] [--mode hybrid|rules|llm] [--model file.gguf]
//                 [--threads N] [--gpu-layers N] [--out prefix] [--no-llm]
//                 [--no-review]                       (raw engine, no human queue)
//                 [--cost-review X] [--cost-misroute X] [--cost-missed-urgent X]
//                 [--cost-false-urgent X] [--cost-compute-per-hour X]
//                 [--urgent-threshold N] [--urgent-review-margin N] [--urgent-guard N]
//  Writes <prefix>_report.md, <prefix>_results.csv and <prefix>_metrics.json
//
//  TO ADD A NEW CATEGORY:  (1) add a row to CATS below  (2) add its keywords to
//  KW_DEFS using its row index  (3) optionally add few-shot examples.  The LLM prompt,
//  answer letters, urgency policy, review routing, cost model, confusion matrices and
//  report all follow from the CATS table automatically.
// =============================================================================
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <array>
#include <map>
#include <algorithm>
#include <cctype>
#include <cmath>
#include <chrono>
#include <thread>
#include <iomanip>
#include <ctime>
#include <cstdio>
#include <cstring>
#include <functional>

#if defined(__has_include)
#  if __has_include("llama.h") && !defined(NO_LLAMA) && !defined(TEST_FAKE_LLM)
#    define USE_LLAMA
#  endif
#endif
#ifdef USE_LLAMA
#  include "llama.h"
#endif

using Clock = std::chrono::steady_clock;
static double msSince(Clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}

// -----------------------------------------------------------------------------
//  1. CATEGORY REGISTRY  (single source of truth - add new categories here)
//     AlwaysUrgent / AlwaysStandard / ContextDependent decides the urgency policy.
//     prefix = incident_id prefix, used as the ground-truth label when evaluating.
// -----------------------------------------------------------------------------
enum class Policy { AlwaysUrgent, AlwaysStandard, ContextDependent };
struct CategoryDef { const char* prefix; const char* tag; char letter; Policy policy; const char* hint; };

static const std::vector<CategoryDef> CATS = {
    {"THF", "Theft/Security",         'A', Policy::AlwaysUrgent,     "stolen cargo, tampered seals, break-ins, hijack attempts, fraud"},
    {"CLD", "Cold Chain/Temperature", 'B', Policy::AlwaysUrgent,     "reefer or cold-room failure, temperature excursions"},
    {"DLY", "Delay/Transit",          'C', Policy::ContextDependent, "late deliveries, traffic, breakdowns, road closures, customs"},
    {"MSP", "Mispick/Order Error",    'D', Policy::AlwaysStandard,   "wrong item, size, quantity or customer in the order"},
    {"DMG", "Damage/Handling",        'E', Policy::AlwaysStandard,   "crushed, torn, wet, dented or broken packaging"},
};
static const int NCAT = (int)CATS.size();
enum Cat { THF = 0, CLD = 1, DLY = 2, MSP = 3, DMG = 4 };   // readable names for the keyword table only

// Extra guidance appended to the LLM system prompt (edit freely).
[[maybe_unused]] static const char* LLM_EXTRA_GUIDANCE =
    "Choose Theft/Security only when there is evidence of tampering, unauthorised access or foul play. "
    "A plain count or item mismatch with no such evidence is an order error.";

// Few-shot examples (prefix, text). They live in the cached prompt prefix, so they cost
// nothing per incident. Keep them different from your test data.
struct Shot { const char* prefix; const char* text; };
static const std::vector<Shot> FEWSHOT = {
    {"THF", "Rear padlock was cut during the night stop and 12 cartons of tablets are gone. The seal number on the paperwork does not match the one on the door."},
    {"THF", "Delivery arrived short by 9 cartons and the pallet wrap had been cut and re-taped. The customer suspects pilferage in transit."},
    {"CLD", "Reefer display shows 11 degrees and rising with a load of frozen prawns. The compressor stopped near the border."},
    {"CLD", "The vaccine cooler data logger exceeded 8 degrees for three hours after the courier left it in a hot van."},
    {"DLY", "Truck is four hours behind after a road closure following an accident. The customer window is tomorrow so there is no rush."},
    {"DLY", "The ferry was cancelled and the dialysis supplies for the clinic are due today."},
    {"MSP", "Customer received 20 units in size M although the order and packing slip say size L."},
    {"MSP", "We counted 40 cartons against 48 on the pick list with no sign of tampering. The picker stopped early."},
    {"DMG", "Bottom layer cartons were crushed under another pallet and two have torn corners. The goods inside look intact."},
    {"DMG", "Shrink wrap was torn and several boxes arrived wet after rain with smeared labels."},
};

struct Incident { std::string id, text, urgency; };

// -----------------------------------------------------------------------------
//  2. KEYWORD TABLES.  Patterns are lowercase words separated by single spaces
//  (punctuation is normalised to spaces, so "break-in" is written "break in").
//  A trailing '*' means "starts with" (stol* -> stolen, stole).  Everything else
//  must match whole words.  Weight = how strongly the phrase signals the category.
// -----------------------------------------------------------------------------
struct KwDef { int cat; int w; const char* pat; };
static const KwDef KW_DEFS[] = {
    // ---- Theft / Security ----
    {THF,3,"stol*"},{THF,3,"steal*"},{THF,3,"theft"},{THF,3,"thie*"},{THF,3,"hijack*"},{THF,3,"burglar*"},
    {THF,3,"break in"},{THF,3,"broke into"},{THF,3,"broken into"},{THF,3,"pried"},{THF,3,"tamper*"},
    {THF,3,"forged"},{THF,3,"fake"},{THF,3,"spoof*"},{THF,3,"cloned"},{THF,3,"siphon*"},{THF,3,"jammer"},
    {THF,3,"police"},{THF,3,"insider"},{THF,3,"pilfer*"},{THF,3,"unauthori*"},{THF,3,"stranger"},
    {THF,3,"impersonat*"},{THF,3,"fraud*"},{THF,3,"bogus"},{THF,3,"rummag*"},{THF,3,"intruder*"},
    {THF,3,"robb*"},{THF,3,"swiped"},{THF,3,"suspicious"},
    {THF,2,"cut open"},{THF,2,"cut through"},{THF,2,"padlock"},{THF,2,"cctv"},{THF,2,"camera footage"},
    {THF,2,"security"},{THF,2,"followed"},{THF,2,"unhooked"},{THF,2,"drained"},{THF,2,"slashed"},
    {THF,2,"drilled out"},{THF,2,"unrecognized"},{THF,2,"not an employee"},{THF,2,"not in our system"},
    {THF,2,"replacement seal"},{THF,2,"new bolt seal"},{THF,2,"was cut"},{THF,2,"been cut"},{THF,2,"is cut"},
    {THF,2,"seal was broken"},{THF,2,"seal is broken"},{THF,2,"never authorized"},{THF,2,"unaccounted"},
    {THF,2,"vanished"},{THF,2,"disappeared"},{THF,2,"gate log*"},{THF,2,"badge"},{THF,2,"no damage"},
    {THF,3,"door sensor opened"},{THF,3,"door alert"},{THF,2,"phone is switched off"},{THF,2,"phone has been off"},
    {THF,2,"not answering"},{THF,3,"stopped responding"},{THF,2,"never got it"},{THF,2,"lock cut"},
    {THF,3,"cut the tarp"},{THF,3,"climbed"},{THF,3,"repack*"},{THF,3,"replaced with"},{THF,3,"reapplied"},{THF,3,"redone"},{THF,1,"approached"},{THF,2,"flagged me down"},
    {THF,1,"waved"},{THF,1,"hole"},{THF,1,"fence"},{THF,1,"yard"},{THF,1,"overnight"},
    {THF,1,"seal"},{THF,1,"guard"},{THF,1,"gps"},{THF,1,"tracker"},{THF,1,"gate"},{THF,1,"suspect*"},
    {THF,1,"empty"},{THF,1,"gone"},{THF,1,"missing"},{THF,1,"short"},{THF,1,"tarp"},{THF,1,"camera"},
    {THF,1,"removed"},{THF,1,"replaced"},{THF,1,"unknown"},{THF,1,"does not match"},{THF,1,"did not match"},
    {THF,1,"sleeper"},{THF,1,"parked"},{THF,1,"lock*"},

    // ---- Cold chain / temperature ----
    {CLD,3,"reefer"},{CLD,3,"cold chain"},{CLD,3,"cold room"},{CLD,3,"cold storage"},{CLD,3,"refrigerat*"},
    {CLD,3,"fridge"},{CLD,3,"chiller"},{CLD,3,"freezer"},{CLD,3,"thaw*"},{CLD,3,"defrost*"},{CLD,3,"melted"},
    {CLD,3,"melt*"},{CLD,3,"frozen"},{CLD,3,"frost"},{CLD,3,"compressor"},{CLD,3,"logger"},{CLD,3,"gel pack*"},
    {CLD,3,"coolant"},{CLD,3,"cold pack"},{CLD,3,"temperature*"},{CLD,3,"setpoint"},{CLD,3,"set point"},
    {CLD,3,"probe"},{CLD,3,"thermal"},{CLD,3,"thermostat"},{CLD,3,"excursion*"},{CLD,3,"evaporator"},
    {CLD,3,"shore power"},{CLD,3,"precool*"},{CLD,3,"pre cool*"},{CLD,3,"cooling"},{CLD,3,"cooler"},
    {CLD,3,"minus"},{CLD,3,"chilled"},{CLD,3,"ice cream"},{CLD,3,"biologic*"},{CLD,3,"insulin"},
    {CLD,2,"cold"},{CLD,2,"airflow"},{CLD,2,"gasket"},{CLD,2,"condensation"},{CLD,2,"ripening"},{CLD,2,"co2"},
    {CLD,2,"warm"},{CLD,2,"ice"},{CLD,2,"frozen"},{CLD,2,"defrost"},{CLD,2,"refrigerant"},{CLD,2,"vaccine*"},
    {CLD,2,"quarantine*"},{CLD,2,"specimen*"},{CLD,2,"lab"},{CLD,2,"dairy"},{CLD,2,"seafood"},
    {CLD,1,"sensor"},{CLD,1,"humidity"},{CLD,1,"overheat*"},{CLD,1,"fan*"},{CLD,1,"heat"},{CLD,1,"hot"},
    {CLD,1,"pharma*"},{CLD,1,"fresh"},{CLD,1,"breakdown"},{CLD,1,"fuel"},{CLD,1,"battery"},{CLD,1,"unit"},
    {CLD,1,"display"},{CLD,1,"alarm"},{CLD,1,"power"},

    // ---- Delay / transit ----
    {DLY,3,"traffic"},{DLY,3,"delayed"},{DLY,3,"delay*"},{DLY,3,"detour"},{DLY,3,"gridlock"},
    {DLY,3,"roadblock*"},{DLY,3,"roadwork*"},{DLY,3,"road work*"},{DLY,3,"stranded"},{DLY,3,"eta"},
    {DLY,3,"behind schedule"},{DLY,3,"breakdown"},{DLY,3,"broke down"},{DLY,3,"hours of service"},
    {DLY,3,"drive time"},{DLY,3,"relay driver"},{DLY,3,"cutoff"},{DLY,3,"weigh station"},{DLY,3,"landslide"},
    {DLY,3,"customs"},{DLY,3,"ferry"},{DLY,3,"missed"},{DLY,3,"late"},{DLY,3,"backlog"},{DLY,3,"queue"},
    {DLY,3,"transit"},{DLY,3,"accident"},{DLY,3,"jam"},{DLY,3,"jammed"},{DLY,3,"blowout"},{DLY,3,"flood*"},
    {DLY,3,"snowstorm"},{DLY,3,"protest"},{DLY,3,"strike"},{DLY,1,"holiday"},{DLY,1,"carrier"},
    {DLY,2,"stuck"},{DLY,2,"backup"},{DLY,2,"fog"},{DLY,1,"closed"},{DLY,2,"closure"},{DLY,1,"route"},
    {DLY,1,"highway"},{DLY,1,"road"},{DLY,1,"arrive*"},{DLY,1,"arrival"},{DLY,2,"checkpoint"},{DLY,2,"border"},
    {DLY,1,"rain"},{DLY,1,"snow"},{DLY,1,"showers"},{DLY,2,"festival"},{DLY,2,"procession"},{DLY,1,"slot"},
    {DLY,1,"depot"},{DLY,1,"hub"},{DLY,1,"tracking"},{DLY,1,"miles"},{DLY,1,"km"},{DLY,1,"hours"},
    {DLY,1,"minutes"},{DLY,1,"window"},{DLY,1,"tomorrow"},{DLY,1,"slowed"},{DLY,1,"stop"},{DLY,1,"stopped"},
    {DLY,1,"waiting"},{DLY,1,"waited"},{DLY,1,"driver*"},{DLY,1,"trip"},
    {DLY,1,"deliver*"},{DLY,1,"schedule*"},{DLY,1,"shipment"},

    // ---- Warehouse mispicks / order errors ----
    {MSP,3,"mispick*"},{MSP,3,"picker*"},{MSP,3,"pick list"},{MSP,3,"pick sheet"},{MSP,3,"picked"},
    {MSP,3,"picking"},{MSP,3,"pick"},{MSP,3,"instead of"},{MSP,3,"sku"},{MSP,3,"variant"},{MSP,3,"wms"},
    {MSP,3,"short pick"},{MSP,3,"unit of measure"},{MSP,3,"wrong item"},{MSP,3,"mixed up"},{MSP,3,"mix up"},
    {MSP,3,"packing slip"},{MSP,3,"packing list"},{MSP,3,"bin"},{MSP,3,"aisle"},{MSP,3,"swapped"},
    {MSP,3,"mismatch*"},{MSP,3,"incorrect"},{MSP,3,"expiry"},{MSP,3,"exchange"},{MSP,3,"swap"},
    {MSP,3,"ordered"},{MSP,1,"order"},{MSP,1,"orders"},{MSP,1,"item"},{MSP,1,"items"},
    {MSP,2,"wrong"},{MSP,1,"extra"},{MSP,1,"size"},{MSP,1,"color"},{MSP,1,"colour"},{MSP,2,"flavor"},
    {MSP,2,"quantity"},{MSP,1,"model"},{MSP,1,"label*"},{MSP,1,"barcode*"},{MSP,1,"scan*"},{MSP,1,"correct"},
    {MSP,1,"replacement"},{MSP,1,"refund"},{MSP,1,"credit"},{MSP,1,"edition"},{MSP,1,"version"},
    {MSP,1,"shortage"},{MSP,1,"count"},{MSP,1,"counted"},{MSP,1,"customer"},{MSP,1,"shelf"},{MSP,1,"stock"},
    {MSP,1,"batch"},{MSP,1,"lot"},{MSP,1,"kit"},{MSP,1,"units"},{MSP,1,"pack"},
    {MSP,1,"missing"},{MSP,1,"received"},{MSP,1,"short"},{MSP,1,"returned"},

    // ---- Damaged packaging ----
    {DMG,3,"crush*"},{DMG,3,"dent*"},{DMG,3,"torn"},{DMG,3,"ripped"},{DMG,3,"punctur*"},{DMG,3,"crack*"},
    {DMG,3,"shatter*"},{DMG,3,"chipped"},{DMG,3,"scuff*"},{DMG,3,"bent"},{DMG,3,"squash*"},{DMG,3,"flatten*"},
    {DMG,3,"unsellable"},{DMG,3,"burst"},{DMG,3,"bowed"},{DMG,3,"warped"},{DMG,3,"sagging"},{DMG,3,"cushion*"},
    {DMG,3,"void fill"},{DMG,3,"water stain*"},{DMG,1,"shrink wrap"},{DMG,1,"stretch wrap"},{DMG,3,"packaging"},
    {DMG,3,"damage*"},{DMG,3,"soaked"},{DMG,3,"wet"},{DMG,3,"damp*"},{DMG,3,"moisture"},{DMG,3,"leak*"},
    {DMG,3,"tore"},{DMG,3,"split"},{DMG,3,"spill*"},{DMG,3,"collapsed"},{DMG,3,"dropped"},{DMG,3,"tipped"},
    {DMG,3,"fell"},{DMG,1,"foam"},{DMG,3,"strapping"},{DMG,1,"carton*"},{DMG,1,"cartons"},{DMG,1,"pallet*"},
    {DMG,2,"broken"},{DMG,2,"scratch*"},{DMG,2,"stain*"},{DMG,1,"wrap"},{DMG,1,"corner*"},{DMG,1,"straps"},
    {DMG,1,"forklift"},{DMG,1,"fragile"},{DMG,1,"padding"},{DMG,1,"flaps"},{DMG,1,"tape"},{DMG,1,"stack*"},
    {DMG,1,"dust*"},{DMG,1,"lean*"},{DMG,1,"box"},{DMG,1,"boxes"},{DMG,1,"crates"},{DMG,1,"crate"},
    {DMG,1,"glass"},{DMG,1,"bottles"},{DMG,1,"sacks"},{DMG,1,"bags"},{DMG,1,"drums"},{DMG,1,"drum"},
    {DMG,1,"unloading"},{DMG,1,"unloaded"},{DMG,1,"opened"},{DMG,1,"dirty"},{DMG,1,"grease"},{DMG,1,"oil"},
    {DMG,1,"loose"},{DMG,1,"lid"},{DMG,1,"seal"},{DMG,1,"wooden"},{DMG,1,"skid"},{DMG,1,"load"},
};

// Urgency cues, used for ContextDependent categories.  net = urgent - standard.
struct UrgDef { int w; const char* pat; };
static const UrgDef URG_UP[] = {
    {3,"vaccine*"},{3,"medical"},{3,"medicine*"},{3,"hospital"},{3,"clinic"},{3,"pharmacy"},{3,"patient*"},
    {3,"surgical"},{3,"surgery"},{3,"dialysis"},{3,"organ"},{3,"donor"},{3,"insulin"},{3,"emergency"},
    {3,"penalty"},{3,"penalties"},{3,"late fee"},{3,"chargeback"},{3,"production line"},{3,"assembly line"},
    {3,"line will stop"},{3,"will stop"},{3,"shut down"},{3,"shutdown"},{3,"grounded"},{3,"critical"},
    {3,"time critical"},{3,"time sensitive"},{3,"perishable*"},{3,"shelf life"},{3,"school"},{1,"festival"},{2,"event"},{3,"need another"},
    {3,"wedding"},{3,"concert"},{3,"exhibition"},{3,"launch"},{3,"flight"},{3,"cutoff"},{3,"deadline"},
    {3,"hours of service"},{3,"drive time"},{3,"relay driver"},{3,"urgent*"},{3,"asap"},{3,"immediately"},
    {3,"right away"},{3,"at once"},{3,"just in time"},{3,"halted"},{3,"paused"},{3,"no stock"},
    {3,"fresh"},{3,"milk"},{3,"flowers"},{3,"bread"},{3,"fish"},{3,"chocolate"},{3,"frozen"},{3,"dairy"},
    {3,"food"},{3,"plant"},{3,"factory"},{3,"threatening"},{3,"cancel*"},{3,"compensation"},{3,"heavily"},
    {3,"stranded"},{3,"no signal"},{3,"in the sun"},{3,"wait overnight"},{3,"before 6 am"},{3,"can not wait"},
    {3,"cannot wait"},{3,"stop within"},{3,"within three hours"},{3,"within six hours"},{3,"within 24"},
    {2,"contract"},{2,"customs"},{1,"tonight"},{2,"expir*"},{2,"sale"},{2,"promotion"},{2,"retailer"},
    {2,"failing"},{2,"tow"},{2,"roadside"},{2,"hours late"},{2,"day late"},{2,"missed"},
};
static const UrgDef URG_DOWN[] = {
    {4,"flexible"},{4,"not urgent"},{4,"any urgent"},{4,"no urgent"},{4,"not time sensitive"},
    {4,"not set a strict time"},{4,"no impact"},{4,"no problem"},{4,"acceptable"},{4,"re booked"},
    {4,"rebooked"},{4,"regular stock"},{4,"routine"},{4,"general merchandise"},{4,"no complaints"},
    {4,"complained"},{4,"does not need"},{4,"day after tomorrow"},{4,"one day delay"},{4,"moderate"},
    {4,"light rain"},{4,"fine"},{4,"agreed"},{4,"plenty"},{4,"notified"},{4,"been informed"},{4,"any time"},{4,"when possible"},
    {4,"until noon"},{4,"called ahead"},{4,"should still"},{4,"will still"},{4,"minor"},{4,"showers"},
    {4,"within 48 hours"},{4,"no strict"},{4,"not strict"},{4,"replenishment"},{4,"re book*"},{4,"re scheduled"},
    {4,"rescheduled"},{4,"moved from"},{4,"updated the eta"},{4,"clear it by tomorrow"},{4,"is acceptable"},
    {4,"closed after"},{4,"no one has"},{4,"can unload"},{4,"just like to know"},{4,"stationery"},
    {2,"slightly"},{2,"about an hour"},{2,"an hour"},{2,"tomorrow morning"},{2,"next day"},{2,"a day"},
};

// -----------------------------------------------------------------------------
//  3. RUN CONFIGURATION (all overridable from the command line)
// -----------------------------------------------------------------------------
struct Config {
    std::string csv = "logistics_incident_reports.csv";
    std::string model = "models/qwen2.5-7b-instruct.Q4_K_M.gguf";
    std::string out = "triage";
    std::string mode = "hybrid";          // hybrid | rules | llm
    int threads = 4, gpuLayers = 0;
    bool review = true;                   // route doubtful cases to the human queue
    // --- cost model (money per event; PLACEHOLDERS - set to your real figures) ---
    double costReview       = 0.50;       // one human review of one incident
    double costMisroute     = 2.00;       // wrong queue -> rework / delay
    double costMissedUrgent = 50.00;      // EXTRA cost when an urgent case is routed as standard
    double costFalseUrgent  = 1.00;       // EXTRA cost when a standard case is flagged urgent
    double computePerHour   = 0.10;       // machine cost per hour of engine run time
    // --- review / urgency knobs ---
    int urgentThreshold = 2;              // net urgency score >= this -> Urgent (ContextDependent cats)
    int urgentReviewMargin = 2;           // predicted Standard but within this many points -> human
    int urgentGuard = 3;                  // predicted non-urgent but urgent-category evidence >= this -> human
};
static Config cfg;

// Rules are only trusted when the winning score is high enough AND clearly ahead.
static const int MIN_TOP = 5;
static const int MIN_MARGIN = 2;

// -----------------------------------------------------------------------------
//  4. Text normalisation + compiled matchers
// -----------------------------------------------------------------------------
static std::string normalize(const std::string& s) {
    std::string out;
    out.reserve(s.size() + 2);
    out.push_back(' ');
    bool lastSpace = true;
    for (unsigned char c : s) {
        if (std::isalnum(c)) { out.push_back((char)std::tolower(c)); lastSpace = false; }
        else if (!lastSpace) { out.push_back(' '); lastSpace = true; }
    }
    if (!lastSpace) out.push_back(' ');
    return out;
}

struct Matcher { int cat; int w; std::string needle; std::string label; };

static Matcher makeMatcher(int cat, int w, const char* pat) {
    std::string p = pat, needle;
    if (!p.empty() && p.back() == '*') needle = " " + p.substr(0, p.size() - 1);
    else needle = " " + p + " ";
    return {cat, w, needle, p};
}
static std::vector<Matcher> buildMatchers() {
    std::vector<Matcher> m;
    for (const auto& k : KW_DEFS) m.push_back(makeMatcher(k.cat, k.w, k.pat));
    return m;
}
static std::vector<Matcher> buildUrg(const UrgDef* a, size_t n) {
    std::vector<Matcher> m;
    for (size_t i = 0; i < n; ++i) m.push_back(makeMatcher(0, a[i].w, a[i].pat));
    return m;
}

struct RuleResult {
    int cat = -1;                 // best category (-1 = no keyword hit)
    int top = 0, margin = 0;      // best score, lead over runner-up
    std::vector<int> score;       // per-category score
    std::vector<int> hits;        // indices into the matcher table
};

static RuleResult ruleClassify(const std::string& norm, const std::vector<Matcher>& ms) {
    RuleResult r;
    r.score.assign(NCAT, 0);
    for (size_t i = 0; i < ms.size(); ++i) {
        if (norm.find(ms[i].needle) != std::string::npos) {
            r.score[ms[i].cat] += ms[i].w;
            r.hits.push_back((int)i);
        }
    }
    // Damage needs at least one real damage word (weight >= 2). Packaging nouns alone
    // (carton, pallet, load, wrap...) also show up in theft and tampering reports.
    bool dmgReal = false;
    for (int h : r.hits) if (ms[h].cat == DMG && ms[h].w >= 2) { dmgReal = true; break; }
    if (!dmgReal) r.score[DMG] = 0;

    int best = -1, bestS = 0, second = 0;
    for (int c = 0; c < NCAT; ++c) {
        if (r.score[c] > bestS) { second = bestS; bestS = r.score[c]; best = c; }
        else if (r.score[c] > second) second = r.score[c];
    }
    r.cat = best; r.top = bestS; r.margin = bestS - second;
    return r;
}

static int urgencyNet(const std::string& norm, const std::vector<Matcher>& up, const std::vector<Matcher>& down) {
    int net = 0;
    for (const auto& m : up)   if (norm.find(m.needle) != std::string::npos) net += m.w;
    for (const auto& m : down) if (norm.find(m.needle) != std::string::npos) net -= m.w;
    return net;
}

// -----------------------------------------------------------------------------
//  5. Cost model helpers
// -----------------------------------------------------------------------------
// Cost of routing an incident whose TRUE category is `truth` to `pred` (category-level proxy,
// used to decide whether an LLM answer is risky enough to need a human).
static double pairCost(int truth, int pred) {
    if (truth == pred) return 0.0;
    double c = cfg.costMisroute;
    Policy a = CATS[truth].policy, b = CATS[pred].policy;
    if (a == Policy::AlwaysUrgent && b != Policy::AlwaysUrgent) c += cfg.costMissedUrgent;   // conservative
    if (a == Policy::AlwaysStandard && b == Policy::AlwaysUrgent) c += cfg.costFalseUrgent;
    return c;
}
static double expectedCost(int pred, const std::vector<float>& p) {
    double ec = 0;
    for (int j = 0; j < NCAT && j < (int)p.size(); ++j) if (j != pred) ec += (double)p[j] * pairCost(j, pred);
    return ec;
}
// Realised cost of an automatic (not human-reviewed) decision, when ground truth is known.
static double realisedCost(int truthCat, int truthUrg, int predCat, bool predUrgent, double* catPart, double* urgPart) {
    double c = 0, u = 0;
    if (truthCat >= 0 && predCat != truthCat) c = cfg.costMisroute;
    if (truthUrg == 0 && !predUrgent) u = cfg.costMissedUrgent;
    else if (truthUrg == 1 && predUrgent) u = cfg.costFalseUrgent;
    if (catPart) *catPart = c;
    if (urgPart) *urgPart = u;
    return c + u;
}

// -----------------------------------------------------------------------------
//  6. CSV loading
// -----------------------------------------------------------------------------
static std::vector<std::string> parseCSVLine(const std::string& line) {
    std::vector<std::string> tok;
    std::string cur;
    bool q = false;
    for (size_t i = 0; i < line.size(); ++i) {
        char c = line[i];
        if (c == '"') {
            if (q && i + 1 < line.size() && line[i + 1] == '"') { cur.push_back('"'); ++i; }
            else q = !q;
        } else if (c == ',' && !q) { tok.push_back(cur); cur.clear(); }
        else cur.push_back(c);
    }
    tok.push_back(cur);
    return tok;
}
static std::vector<Incident> loadIncidents(const std::string& path) {
    std::vector<Incident> v;
    std::ifstream f(path);
    if (!f.is_open()) { std::cerr << "Error: could not open " << path << "\n"; return v; }
    std::string line;
    std::getline(f, line); // header
    while (std::getline(f, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line.empty()) continue;
        auto t = parseCSVLine(line);
        if (t.size() >= 3) v.push_back({t[0], t[1], t[2]});
    }
    return v;
}

// -----------------------------------------------------------------------------
//  7. LOCAL LLM:  ONE forward pass per incident, no text generation.
//   - system prompt + few-shot examples are decoded once and kept in the KV cache
//   - each incident only decodes its own tokens
//   - we softmax the logits of the answer letters -> a probability per category
// -----------------------------------------------------------------------------
[[maybe_unused]] static void softmaxInPlace(std::vector<float>& v) {
    if (v.empty()) return;
    float mx = *std::max_element(v.begin(), v.end());
    double s = 0;
    for (float& x : v) { x = std::exp(x - mx); s += x; }
    for (float& x : v) x = (float)(x / s);
}

#if defined(USE_LLAMA)
class LocalLLM {
public:
    ~LocalLLM() {
        if (ctx) llama_free(ctx);
        if (model) llama_model_free(model);
        if (backendUp) llama_backend_free();
    }

    bool init(const std::string& path, int threads, int gpuLayers, std::string& err) {
        if (NCAT > 26) { err = "too many categories for single-letter answers"; return false; }
        llama_log_set([](ggml_log_level lvl, const char* text, void*) {
            if (lvl == GGML_LOG_LEVEL_ERROR) std::fputs(text, stderr);
        }, nullptr);
        llama_backend_init();
        backendUp = true;

        llama_model_params mp = llama_model_default_params();
        mp.n_gpu_layers = gpuLayers;
        model = llama_model_load_from_file(path.c_str(), mp);
        if (!model) { err = "could not load model file: " + path; return false; }
        vocab = llama_model_get_vocab(model);

        llama_context_params cp = llama_context_default_params();
        cp.n_ctx = 2048;
        cp.n_batch = 512;
        cp.n_ubatch = 512;
        cp.n_threads = threads;
        cp.n_threads_batch = threads;
        ctx = llama_init_from_model(model, cp);
        if (!ctx) { err = "could not create llama context"; return false; }

        label.assign(NCAT, 0);
        for (int i = 0; i < NCAT; ++i) {
            auto t = tok(std::string(1, CATS[i].letter), false, false);
            if (t.size() != 1) { err = "answer letters are not single tokens for this model"; return false; }
            label[i] = t[0];
        }

        // ChatML template (Qwen / many others). Change this block for other model families.
        std::string letters;
        prefixText = "<|im_start|>system\nYou are an expert logistics incident triage AI. "
                     "Read the incident report and choose the single best category:\n";
        for (int i = 0; i < NCAT; ++i) {
            prefixText += std::string(1, CATS[i].letter) + " = " + CATS[i].tag + " (" + CATS[i].hint + ")\n";
            letters += (i ? (i + 1 == NCAT ? " or " : ", ") : "") + std::string(1, CATS[i].letter);
        }
        prefixText += std::string(LLM_EXTRA_GUIDANCE) + "\nReply with exactly one letter: " + letters + ".<|im_end|>\n";
        for (const auto& s : FEWSHOT) {
            char letter = 0;
            for (const auto& c : CATS) if (std::string(c.prefix) == s.prefix) letter = c.letter;
            if (!letter) continue;
            prefixText += std::string("<|im_start|>user\n") + s.text + "<|im_end|>\n<|im_start|>assistant\n" + letter + "<|im_end|>\n";
        }
        prefixText += "<|im_start|>user\n";
        return decodePrefix(err);
    }

    // Fills probs (size NCAT) and returns true on success.
    bool classify(const std::string& incident, std::vector<float>& probs) {
        llama_memory_t mem = llama_get_memory(ctx);
        if (!llama_memory_seq_rm(mem, 0, prefixLen, -1)) {      // rewind to the cached prefix
            llama_memory_clear(mem, true);
            std::string e;
            if (!decodePrefix(e)) return false;
        }
        std::string clean = incident.substr(0, 700);
        for (size_t p; (p = clean.find("<|")) != std::string::npos;) clean.replace(p, 2, "< ");
        auto t = tok(clean + "<|im_end|>\n<|im_start|>assistant\n", false, true);
        if (t.empty()) return false;
        llama_batch b = llama_batch_get_one(t.data(), (int32_t)t.size());
        if (llama_decode(ctx, b) != 0) return false;
        const float* lg = llama_get_logits_ith(ctx, -1);
        if (!lg) return false;
        probs.assign(NCAT, 0.f);
        for (int i = 0; i < NCAT; ++i) probs[i] = lg[label[i]];
        softmaxInPlace(probs);
        return true;
    }

private:
    std::vector<llama_token> tok(const std::string& s, bool addSpecial, bool parseSpecial) {
        std::vector<llama_token> t(s.size() + 16);
        int n = llama_tokenize(vocab, s.c_str(), (int32_t)s.size(), t.data(), (int32_t)t.size(), addSpecial, parseSpecial);
        if (n < 0) {
            t.resize(-n);
            n = llama_tokenize(vocab, s.c_str(), (int32_t)s.size(), t.data(), (int32_t)t.size(), addSpecial, parseSpecial);
        }
        t.resize(n > 0 ? n : 0);
        return t;
    }
    bool decodePrefix(std::string& err) {
        auto pt = tok(prefixText, true, true);
        if (pt.empty()) { err = "prefix tokenization failed"; return false; }
        for (size_t off = 0; off < pt.size(); off += 512) {          // respect n_batch
            int32_t n = (int32_t)std::min<size_t>(512, pt.size() - off);
            llama_batch b = llama_batch_get_one(pt.data() + off, n);
            if (llama_decode(ctx, b) != 0) { err = "prefix decode failed"; return false; }
        }
        prefixLen = (int)pt.size();
        return true;
    }

    llama_model* model = nullptr;
    llama_context* ctx = nullptr;
    const llama_vocab* vocab = nullptr;
    bool backendUp = false;
    int prefixLen = 0;
    std::vector<llama_token> label;
    std::string prefixText;
};
#elif defined(TEST_FAKE_LLM)
// Wiring test only: deterministic pseudo-answers so the routing/report code can be exercised.
class LocalLLM {
public:
    bool init(const std::string&, int, int, std::string&) { return true; }
    bool classify(const std::string& t, std::vector<float>& p) {
        size_t h = std::hash<std::string>{}(t);
        int best = (int)(h % NCAT);
        float conf = 0.45f + 0.5f * (float)((h >> 8) % 100) / 100.0f;
        p.assign(NCAT, (1.0f - conf) / (NCAT - 1));
        p[best] = conf;
        return true;
    }
};
#else
class LocalLLM {
public:
    bool init(const std::string&, int, int, std::string& err) { err = "built without llama.cpp (llama.h not found)"; return false; }
    bool classify(const std::string&, std::vector<float>&) { return false; }
};
#endif

// -----------------------------------------------------------------------------
//  8. Small helpers
// -----------------------------------------------------------------------------
static std::string truncUtf8(const std::string& s, size_t n) {
    if (s.size() <= n) return s;
    size_t cut = n > 3 ? n - 3 : 0;
    while (cut > 0 && ((unsigned char)s[cut] & 0xC0) == 0x80) --cut;
    return s.substr(0, cut) + "...";
}
static std::string mdEsc(std::string s) { for (char& c : s) if (c == '|' || c == '\n' || c == '\r') c = ' '; return s; }
static std::string csvQuote(const std::string& s) {
    std::string o = "\"";
    for (char c : s) { if (c == '"') o += "\"\""; else o += c; }
    return o + "\"";
}
static std::string num(double v, int p = 2) { std::ostringstream o; o << std::fixed << std::setprecision(p) << v; return o.str(); }
static std::string pct(double a, double b) { return b <= 0 ? "n/a" : num(100.0 * a / b, 1) + "%"; }
static std::string money(double v) { return "$" + num(v, 2); }
static double percentile(std::vector<double> v, double p) {
    if (v.empty()) return 0;
    std::sort(v.begin(), v.end());
    size_t k = (size_t)std::ceil(p * v.size());
    if (k == 0) k = 1;
    return v[std::min(k, v.size()) - 1];
}
static double mean(const std::vector<double>& v) {
    if (v.empty()) return 0;
    double s = 0; for (double x : v) s += x; return s / v.size();
}
static int truthCatFromId(const std::string& id) {
    for (int c = 0; c < NCAT; ++c) if (id.rfind(CATS[c].prefix, 0) == 0) return c;
    return -1;
}
static int truthUrgency(std::string u) {
    for (auto& c : u) c = (char)std::tolower((unsigned char)c);
    if (u.find("urgent") != std::string::npos) return 0;
    if (u.find("standard") != std::string::npos) return 1;
    return -1;
}
static const char* catName(int c) { return (c >= 0 && c < NCAT) ? CATS[c].tag : "Unclassified"; }
static const char* catPrefix(int c) { return (c >= 0 && c < NCAT) ? CATS[c].prefix : "none"; }

struct Result {
    int truthCat = -1, truthUrg = -1;
    int autoCat = -1; bool autoUrgent = true;      // the engine's own answer (before any human)
    bool review = false; std::string reviewReason;
    std::string source;                            // rules | llm | rules-lowconf | unclassified
    int ruleCat = -1, ruleTop = 0, ruleMargin = 0, urgNet = 0;
    int llmCat = -1; float llmConf = 0; double llmMs = 0;
    double latencyMs = 0;
    std::string hits;
    // costs (money)
    double qualityCost = 0, reviewCost = 0;
};

static Config parseArgs(int argc, char** argv) {
    Config o;
    unsigned hw = std::thread::hardware_concurrency();
    o.threads = hw ? (int)hw : 4;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto str = [&](std::string& dst) { if (i + 1 < argc) dst = argv[++i]; };
        auto dbl = [&](double& dst) { if (i + 1 < argc) dst = std::atof(argv[++i]); };
        auto integer = [&](int& dst) { if (i + 1 < argc) dst = std::atoi(argv[++i]); };
        if (a == "--csv") str(o.csv);
        else if (a == "--model") str(o.model);
        else if (a == "--out") str(o.out);
        else if (a == "--mode") str(o.mode);
        else if (a == "--threads") { integer(o.threads); o.threads = std::max(1, o.threads); }
        else if (a == "--gpu-layers") { integer(o.gpuLayers); o.gpuLayers = std::max(0, o.gpuLayers); }
        else if (a == "--no-llm") o.mode = "rules";
        else if (a == "--no-review") o.review = false;
        else if (a == "--cost-review") dbl(o.costReview);
        else if (a == "--cost-misroute") dbl(o.costMisroute);
        else if (a == "--cost-missed-urgent") dbl(o.costMissedUrgent);
        else if (a == "--cost-false-urgent") dbl(o.costFalseUrgent);
        else if (a == "--cost-compute-per-hour") dbl(o.computePerHour);
        else if (a == "--urgent-threshold") integer(o.urgentThreshold);
        else if (a == "--urgent-review-margin") integer(o.urgentReviewMargin);
        else if (a == "--urgent-guard") integer(o.urgentGuard);
    }
    return o;
}

// -----------------------------------------------------------------------------
//  9. main
// -----------------------------------------------------------------------------
int main(int argc, char** argv) {
    cfg = parseArgs(argc, argv);
    if (cfg.mode != "hybrid" && cfg.mode != "rules" && cfg.mode != "llm") {
        std::cerr << "Unknown --mode '" << cfg.mode << "' (use hybrid, rules or llm)\n";
        return 2;
    }
    for (const auto& k : KW_DEFS)
        if (k.cat < 0 || k.cat >= NCAT) { std::cerr << "Keyword table refers to a category that is not in CATS: " << k.pat << "\n"; return 2; }

    std::cout << "==================================================\n"
              << "   HYBRID INCIDENT TRIAGE ENGINE v2  (mode: " << cfg.mode << ")\n"
              << "==================================================\n";

    std::vector<Incident> incidents = loadIncidents(cfg.csv);
    if (incidents.empty()) { std::cout << "No incidents loaded from '" << cfg.csv << "'.\n"; return 1; }
    std::cout << "Loaded " << incidents.size() << " incidents from " << cfg.csv << "\n";

    // ---- optional LLM -------------------------------------------------------
    LocalLLM llm;
    bool llmReady = false;
    std::string llmNote;
    double loadMs = 0;
    if (cfg.mode != "rules") {
        auto t0 = Clock::now();
        std::string err;
        llmReady = llm.init(cfg.model, cfg.threads, cfg.gpuLayers, err);
        loadMs = msSince(t0);
        llmNote = llmReady ? ("loaded " + cfg.model + " in " + num(loadMs / 1000.0) + " s") : ("NOT available: " + err);
        if (!llmReady && cfg.mode == "llm") { std::cerr << "LLM-only mode needs a working model: " << llmNote << "\n"; return 2; }
    } else {
        llmNote = "disabled (rules mode)";
    }
    std::cout << "LLM: " << llmNote << "\n\n";

    // ---- compile matchers once ---------------------------------------------
    const std::vector<Matcher> kw   = buildMatchers();
    const std::vector<Matcher> up   = buildUrg(URG_UP,   sizeof(URG_UP)   / sizeof(URG_UP[0]));
    const std::vector<Matcher> down = buildUrg(URG_DOWN, sizeof(URG_DOWN) / sizeof(URG_DOWN[0]));

    // ---- classify -----------------------------------------------------------
    std::vector<Result> res(incidents.size());
    std::vector<std::vector<float>> allProbs(incidents.size());
    long long ruleNs = 0;
    int llmCalls = 0;
    auto wall0 = Clock::now();

    for (size_t i = 0; i < incidents.size(); ++i) {
        const Incident& inc = incidents[i];
        Result& r = res[i];
        auto tIn = Clock::now();
        r.truthCat = truthCatFromId(inc.id);
        r.truthUrg = truthUrgency(inc.urgency);

        auto t0 = Clock::now();
        std::string norm = normalize(inc.text);
        RuleResult rr = ruleClassify(norm, kw);
        ruleNs += std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - t0).count();
        r.ruleCat = rr.cat; r.ruleTop = rr.top; r.ruleMargin = rr.margin;
        const bool ruleConfident = rr.cat >= 0 && rr.top >= MIN_TOP && rr.margin >= MIN_MARGIN;

        // ---- category decision ----
        std::vector<float> probs;
        bool useLLM = llmReady && (cfg.mode == "llm" || (cfg.mode == "hybrid" && !ruleConfident));
        if (useLLM) {
            auto l0 = Clock::now();
            bool ok = llm.classify(inc.text, probs);
            r.llmMs = msSince(l0);
            if (ok && (int)probs.size() == NCAT) {
                int best = (int)(std::max_element(probs.begin(), probs.end()) - probs.begin());
                r.autoCat = best; r.llmCat = best; r.llmConf = probs[best]; r.source = "llm";
                allProbs[i] = probs; ++llmCalls;
            }
        }
        if (r.autoCat < 0 && cfg.mode != "llm") {
            if (ruleConfident)      { r.autoCat = rr.cat; r.source = "rules"; }
            else if (rr.cat >= 0)   { r.autoCat = rr.cat; r.source = "rules-lowconf"; }
        }
        if (r.autoCat < 0) r.source = "unclassified";

        // ---- urgency decision (policy per category; fail-safe = Urgent) ----
        r.autoUrgent = true;
        if (r.autoCat >= 0) {
            Policy pol = CATS[r.autoCat].policy;
            if (pol == Policy::AlwaysUrgent) r.autoUrgent = true;
            else if (pol == Policy::AlwaysStandard) r.autoUrgent = false;
            else { r.urgNet = urgencyNet(norm, up, down); r.autoUrgent = r.urgNet >= cfg.urgentThreshold; }
        }

        // ---- human review routing ----
        if (cfg.review) {
            if (r.autoCat < 0) r.reviewReason = "no-category-signal";
            else if (r.source == "rules-lowconf") r.reviewReason = "low-confidence-rules";
            else if (r.source == "llm" && expectedCost(r.autoCat, probs) > cfg.costReview) r.reviewReason = "llm-expected-cost";
            if (r.reviewReason.empty() && r.autoCat >= 0 && CATS[r.autoCat].policy != Policy::AlwaysUrgent) {
                for (int c = 0; c < NCAT; ++c)
                    if (CATS[c].policy == Policy::AlwaysUrgent && rr.score[c] >= cfg.urgentGuard) { r.reviewReason = "urgent-evidence"; break; }
            }
            if (r.reviewReason.empty() && r.autoCat >= 0 && CATS[r.autoCat].policy == Policy::ContextDependent &&
                !r.autoUrgent && r.urgNet >= cfg.urgentThreshold - cfg.urgentReviewMargin) r.reviewReason = "borderline-urgency";
            r.review = !r.reviewReason.empty();
        }

        for (int h : rr.hits) if (kw[h].cat == r.autoCat) { if (!r.hits.empty()) r.hits += "; "; r.hits += kw[h].label; }

        // ---- costs ----
        if (r.review) r.reviewCost = cfg.costReview;
        else if (r.truthCat >= 0 || r.truthUrg >= 0) r.qualityCost = realisedCost(r.truthCat, r.truthUrg, r.autoCat, r.autoUrgent, nullptr, nullptr);
        r.latencyMs = msSince(tIn);
    }
    const double wallMs = msSince(wall0);

    // ==================== statistics ====================
    const size_t N = incidents.size();
    std::vector<std::vector<int>> conf(NCAT, std::vector<int>(NCAT + 1, 0));   // raw engine answers
    int uConf[2][2] = {{0, 0}, {0, 0}};                                        // raw engine urgency
    int catKnown = 0, catCorrectRaw = 0, urgKnown = 0, urgCorrectRaw = 0;
    int autoN = 0, autoCatCorrect = 0, autoKnown = 0, missedUrgAuto = 0, missedUrgRaw = 0, falseUrgAuto = 0;
    int reviewN = 0, caught = 0, unneeded = 0, residualWrong = 0;
    std::map<std::string, int> srcN, srcKnown, srcCorrect, reasonN, reasonCaught;
    std::vector<int> uTot(NCAT, 0), uOk(NCAT, 0);
    std::vector<std::map<std::string, int>> kwCount(NCAT);
    std::vector<double> latAll, latRules, latLLM;
    double qualityCostTotal = 0, reviewCostTotal = 0, catCostTotal = 0, urgCostTotal = 0;

    for (size_t i = 0; i < N; ++i) {
        const Result& r = res[i];
        srcN[r.source]++;
        latAll.push_back(r.latencyMs);
        (r.source == "llm" ? latLLM : latRules).push_back(r.latencyMs);
        if (r.truthCat >= 0) {
            conf[r.truthCat][r.autoCat >= 0 ? r.autoCat : NCAT]++;
            ++catKnown; ++srcKnown[r.source];
            if (r.autoCat == r.truthCat) { ++catCorrectRaw; ++srcCorrect[r.source]; }
        }
        bool urgOk = true, urgFound = false;
        if (r.truthUrg >= 0) {
            urgFound = true;
            uConf[r.truthUrg][r.autoUrgent ? 0 : 1]++;
            ++urgKnown;
            urgOk = (r.truthUrg == 0) == r.autoUrgent;
            if (urgOk) ++urgCorrectRaw;
            if (r.truthUrg == 0 && !r.autoUrgent) ++missedUrgRaw;
            if (r.truthCat >= 0) { ++uTot[r.truthCat]; if (urgOk) ++uOk[r.truthCat]; }
        }
        bool wrong = (r.truthCat >= 0 && r.autoCat != r.truthCat) || (urgFound && !urgOk);
        if (r.review) {
            ++reviewN; reasonN[r.reviewReason]++;
            if (wrong) { ++caught; reasonCaught[r.reviewReason]++; } else ++unneeded;
        } else {
            ++autoN;
            if (r.truthCat >= 0) { ++autoKnown; if (r.autoCat == r.truthCat) ++autoCatCorrect; }
            if (wrong) ++residualWrong;
            if (r.truthUrg == 0 && !r.autoUrgent) ++missedUrgAuto;
            if (r.truthUrg == 1 && r.autoUrgent) ++falseUrgAuto;
            double cp = 0, up2 = 0;
            realisedCost(r.truthCat, r.truthUrg, r.autoCat, r.autoUrgent, &cp, &up2);
            catCostTotal += cp; urgCostTotal += up2;
        }
        qualityCostTotal += r.qualityCost; reviewCostTotal += r.reviewCost;
        if (r.source == "rules" && r.autoCat >= 0) {
            std::stringstream ss(r.hits); std::string k;
            while (std::getline(ss, k, ';')) { if (!k.empty() && k[0] == ' ') k.erase(0, 1); if (!k.empty()) kwCount[r.autoCat][k]++; }
        }
    }
    const double ruleMs = ruleNs / 1e6;
    double llmMsTotal = 0, llmMsMax = 0;
    for (const auto& r : res) { llmMsTotal += r.llmMs; llmMsMax = std::max(llmMsMax, r.llmMs); }
    const double avgLlm = llmCalls ? llmMsTotal / llmCalls : 0;
    const double computeCost = wallMs / 3.6e6 * cfg.computePerHour;
    const double totalCost = computeCost + reviewCostTotal + qualityCostTotal;
    const int e2eCorrect = autoCatCorrect + reviewN;   // reviewers assumed correct
    const double per1k = 1000.0 / (double)N;

    // ==================== results CSV ====================
    {
        std::ofstream f(cfg.out + "_results.csv");
        f << "incident_id,true_category,engine_category,category_correct,source,llm_confidence,human_review,review_reason,"
             "true_urgency,engine_urgency,urgency_correct,incident_cost,latency_ms,llm_ms,matched_keywords\n";
        for (size_t i = 0; i < N; ++i) {
            const Result& r = res[i];
            bool uok = r.truthUrg >= 0 && ((r.truthUrg == 0) == r.autoUrgent);
            f << incidents[i].id << ',' << csvQuote(r.truthCat >= 0 ? CATS[r.truthCat].tag : "unknown") << ','
              << csvQuote(catName(r.autoCat)) << ',' << (r.truthCat >= 0 ? (r.autoCat == r.truthCat ? "yes" : "no") : "n/a") << ','
              << r.source << ',' << (r.llmCat >= 0 ? num(r.llmConf, 3) : "") << ',' << (r.review ? "yes" : "no") << ',' << r.reviewReason << ','
              << incidents[i].urgency << ',' << (r.autoUrgent ? "Urgent" : "Standard") << ',' << (r.truthUrg >= 0 ? (uok ? "yes" : "no") : "n/a") << ','
              << num(r.qualityCost + r.reviewCost, 2) << ',' << num(r.latencyMs, 3) << ',' << num(r.llmMs, 1) << ',' << csvQuote(r.hits) << '\n';
        }
    }

    // ==================== metrics JSON (for engine-vs-engine comparison) ====================
    {
        std::ofstream j(cfg.out + "_metrics.json");
        j << "{\n"
          << "  \"mode\": \"" << cfg.mode << "\",\n"
          << "  \"model\": \"" << (llmReady ? cfg.model : "") << "\",\n"
          << "  \"incidents\": " << N << ",\n"
          << "  \"human_review_enabled\": " << (cfg.review ? "true" : "false") << ",\n"
          << "  \"quality\": {\n"
          << "    \"raw_category_accuracy\": " << num(catKnown ? (double)catCorrectRaw / catKnown : 0, 4) << ",\n"
          << "    \"raw_urgency_accuracy\": " << num(urgKnown ? (double)urgCorrectRaw / urgKnown : 0, 4) << ",\n"
          << "    \"raw_missed_urgent\": " << missedUrgRaw << ",\n"
          << "    \"automation_coverage\": " << num((double)autoN / N, 4) << ",\n"
          << "    \"auto_handled_category_accuracy\": " << num(autoKnown ? (double)autoCatCorrect / autoKnown : 0, 4) << ",\n"
          << "    \"auto_handled_missed_urgent\": " << missedUrgAuto << ",\n"
          << "    \"auto_handled_false_urgent\": " << falseUrgAuto << ",\n"
          << "    \"human_review_rate\": " << num((double)reviewN / N, 4) << ",\n"
          << "    \"end_to_end_category_accuracy_assuming_correct_reviewers\": " << num(catKnown ? (double)e2eCorrect / catKnown : 0, 4) << "\n"
          << "  },\n"
          << "  \"latency_ms\": {\n"
          << "    \"mean\": " << num(mean(latAll), 4) << ", \"p50\": " << num(percentile(latAll, 0.50), 4)
          << ", \"p95\": " << num(percentile(latAll, 0.95), 4) << ", \"p99\": " << num(percentile(latAll, 0.99), 4)
          << ", \"max\": " << num(percentile(latAll, 1.0), 4) << ",\n"
          << "    \"throughput_per_s\": " << num(wallMs > 0 ? N / (wallMs / 1000.0) : 0, 2) << ",\n"
          << "    \"model_load_s\": " << num(loadMs / 1000.0, 3) << "\n"
          << "  },\n"
          << "  \"cost\": {\n"
          << "    \"compute\": " << num(computeCost, 6) << ", \"human_review\": " << num(reviewCostTotal, 2)
          << ", \"misroute_and_urgency_errors\": " << num(qualityCostTotal, 2) << ", \"total\": " << num(totalCost, 2) << ",\n"
          << "    \"total_per_1000_incidents\": " << num(totalCost * per1k, 2) << "\n"
          << "  }\n}\n";
    }

    // ==================== markdown report ====================
    {
        std::ofstream f(cfg.out + "_report.md");
        std::time_t now = std::time(nullptr);
        char stamp[64]; std::strftime(stamp, sizeof(stamp), "%Y-%m-%d %H:%M:%S", std::localtime(&now));
        const char* srcs[] = {"rules", "llm", "rules-lowconf", "unclassified"};
        const char* srcDesc[] = {"Rules (confident)", "LLM", "Rules (low confidence)", "Unclassified"};
        const size_t CAP = 40;

        f << "# Logistics Incident Triage - Run Report\n\nGenerated: " << stamp << "\n\n";

        // 1 --------------------------------------------------------------------
        f << "## 1. Run summary\n\n| Item | Value |\n|---|---|\n"
          << "| Engine mode | " << cfg.mode << " |\n"
          << "| Dataset | `" << cfg.csv << "` (" << N << " incidents) |\n"
          << "| LLM | " << mdEsc(llmNote) << " |\n"
          << "| CPU threads | " << cfg.threads << " |\n"
          << "| Human review routing | " << (cfg.review ? "ON" : "OFF (raw engine)") << " |\n"
          << "| Rule confidence | top score >= " << MIN_TOP << " and lead >= " << MIN_MARGIN << " |\n"
          << "| Urgency threshold / review margin / urgent-evidence guard | " << cfg.urgentThreshold << " / " << cfg.urgentReviewMargin << " / " << cfg.urgentGuard << " |\n"
          << "| Cost assumptions | review " << money(cfg.costReview) << ", misroute " << money(cfg.costMisroute)
          << ", missed urgent (extra) " << money(cfg.costMissedUrgent) << ", false urgent (extra) " << money(cfg.costFalseUrgent)
          << ", compute " << money(cfg.computePerHour) << "/hour |\n\n";

        // 2 --------------------------------------------------------------------
        f << "## 2. Headline comparison metrics\n\n| Dimension | Metric | Value |\n|---|---|---|\n"
          << "| Quality | Raw category accuracy (engine alone) | " << pct(catCorrectRaw, catKnown) << " |\n"
          << "| Quality | Raw urgent recall (engine alone) | " << pct(uConf[0][0], uConf[0][0] + uConf[0][1]) << " |\n"
          << "| Quality | Auto-handled category accuracy | " << pct(autoCatCorrect, autoKnown) << " |\n"
          << "| Quality | **Missed urgent among auto-handled** | **" << missedUrgAuto << "** |\n"
          << "| Quality | Automation coverage / human review rate | " << pct(autoN, N) << " / " << pct(reviewN, N) << " |\n"
          << "| Latency | Mean / p50 / p95 / p99 per incident | " << num(mean(latAll), 3) << " / " << num(percentile(latAll, .5), 3) << " / "
          << num(percentile(latAll, .95), 3) << " / " << num(percentile(latAll, .99), 3) << " ms |\n"
          << "| Latency | Throughput | " << num(wallMs > 0 ? N / (wallMs / 1000.0) : 0, 1) << " incidents/s |\n"
          << "| Cost | Total cost for this run | " << money(totalCost) << " |\n"
          << "| Cost | Total cost per 1,000 incidents | " << money(totalCost * per1k) << " |\n\n";

        // 3 latency ---------------------------------------------------------------
        f << "## 3. Latency and throughput\n\n| Path | Incidents | Mean ms | p50 ms | p95 ms | p99 ms | Max ms |\n|---|---|---|---|---|---|---|\n";
        auto latRow = [&](const char* name, const std::vector<double>& v) {
            f << "| " << name << " | " << v.size() << " | " << num(mean(v), 3) << " | " << num(percentile(v, .5), 3) << " | "
              << num(percentile(v, .95), 3) << " | " << num(percentile(v, .99), 3) << " | " << num(percentile(v, 1.0), 3) << " |\n";
        };
        latRow("All incidents", latAll);
        if (!latRules.empty()) latRow("Rules path", latRules);
        if (!latLLM.empty()) latRow("LLM path", latLLM);
        f << "\n| Metric | Value |\n|---|---|\n"
          << "| Total classification time | " << num(wallMs) << " ms |\n"
          << "| Throughput | " << num(wallMs > 0 ? N / (wallMs / 1000.0) : 0, 1) << " incidents/s |\n"
          << "| Time in rule engine | " << num(ruleMs, 3) << " ms (" << num(ruleMs * 1000.0 / N, 1) << " us per incident) |\n"
          << "| LLM calls / time in LLM | " << llmCalls << " / " << num(llmMsTotal) << " ms |\n"
          << "| Average / slowest LLM call | " << (llmCalls ? num(avgLlm) + " / " + num(llmMsMax) + " ms" : "n/a") << " |\n"
          << "| Model load time (one-off) | " << (llmReady ? num(loadMs / 1000.0) + " s" : "n/a") << " |\n";
        if (llmCalls && cfg.mode == "hybrid")
            f << "| Estimated time if EVERY incident used the LLM | " << num(avgLlm * N / 1000.0, 1) << " s ("
              << num((avgLlm * N) / std::max(wallMs, 0.001), 1) << "x slower than this hybrid run) |\n";
        f << "\nLatency counts engine time only. Human review turnaround is not included (see section 6).\n\n";

        // 4 routing / review ----------------------------------------------------
        f << "## 4. Routing and human review\n\n| Path | Incidents | Share |\n|---|---|---|\n";
        for (int s = 0; s < 4; ++s) f << "| " << srcDesc[s] << " | " << srcN[srcs[s]] << " | " << pct(srcN[srcs[s]], N) << " |\n";
        f << "\n| Outcome | Incidents | Share |\n|---|---|---|\n"
          << "| Handled automatically | " << autoN << " | " << pct(autoN, N) << " |\n"
          << "| Sent to human review | " << reviewN << " | " << pct(reviewN, N) << " |\n\n";
        if (cfg.review) {
            f << "| Review reason | Incidents | Of which engine would have been wrong |\n|---|---|---|\n";
            for (auto& kv : reasonN) f << "| " << kv.first << " | " << kv.second << " | " << reasonCaught[kv.first] << " |\n";
            f << "\n- **Errors caught by review:** " << caught << "  |  **Unneeded reviews:** " << unneeded
              << "  |  **Review precision:** " << pct(caught, reviewN) << "\n"
              << "- **Errors that slipped through (auto-handled but wrong):** " << residualWrong
              << "  |  **Error catch rate:** " << pct(caught, caught + residualWrong) << "\n"
              << "- Unclassified and low-confidence cases always go to a human; the LLM answer goes to a human when its expected misroute cost exceeds the cost of a review (" << money(cfg.costReview) << ").\n\n";
        } else f << "Human review routing is OFF, so every incident is auto-handled.\n\n";

        // 5 quality ---------------------------------------------------------------
        f << "## 5. Quality\n\n"
          << "| View | Correct | Total | Accuracy |\n|---|---|---|---|\n"
          << "| Raw engine, category (all incidents) | " << catCorrectRaw << " | " << catKnown << " | " << pct(catCorrectRaw, catKnown) << " |\n"
          << "| Raw engine, urgency (all incidents) | " << urgCorrectRaw << " | " << urgKnown << " | " << pct(urgCorrectRaw, urgKnown) << " |\n"
          << "| Auto-handled only, category | " << autoCatCorrect << " | " << autoKnown << " | " << pct(autoCatCorrect, autoKnown) << " |\n"
          << "| End-to-end with humans (reviewers assumed correct) | " << e2eCorrect << " | " << catKnown << " | " << pct(e2eCorrect, catKnown) << " |\n\n"
          << "| Engine path | Correct | Total | Accuracy |\n|---|---|---|---|\n";
        for (int s = 0; s < 4; ++s)
            if (srcKnown[srcs[s]]) f << "| " << srcDesc[s] << " | " << srcCorrect[srcs[s]] << " | " << srcKnown[srcs[s]] << " | " << pct(srcCorrect[srcs[s]], srcKnown[srcs[s]]) << " |\n";

        f << "\n### Per-category precision / recall (raw engine)\n\n| Category | Policy | Support | Precision | Recall | F1 |\n|---|---|---|---|---|---|\n";
        for (int c = 0; c < NCAT; ++c) {
            int sup = 0, predicted = 0, tp = conf[c][c];
            for (int j = 0; j <= NCAT; ++j) sup += conf[c][j];
            for (int i = 0; i < NCAT; ++i) predicted += conf[i][c];
            double p = predicted ? (double)tp / predicted : 0, rc = sup ? (double)tp / sup : 0;
            double f1 = (p + rc) > 0 ? 2 * p * rc / (p + rc) : 0;
            const char* pol = CATS[c].policy == Policy::AlwaysUrgent ? "always urgent" : CATS[c].policy == Policy::AlwaysStandard ? "always standard" : "context";
            f << "| " << CATS[c].tag << " | " << pol << " | " << sup << " | " << pct(tp, predicted) << " | " << pct(tp, sup) << " | " << num(f1, 3) << " |\n";
        }
        f << "\n### Confusion matrix (rows = true, columns = engine answer)\n\n| True \\ Engine |";
        for (int c = 0; c < NCAT; ++c) f << ' ' << CATS[c].prefix << " |";
        f << " Unclassified |\n|---|";
        for (int c = 0; c <= NCAT; ++c) f << "---|";
        f << "\n";
        for (int t = 0; t < NCAT; ++t) {
            f << "| " << CATS[t].tag << " |";
            for (int p = 0; p <= NCAT; ++p) f << ' ' << conf[t][p] << " |";
            f << "\n";
        }
        f << "\n### Urgency (raw engine)\n\n| True \\ Engine | Urgent | Standard |\n|---|---|---|\n"
          << "| Urgent | " << uConf[0][0] << " | " << uConf[0][1] << " |\n"
          << "| Standard | " << uConf[1][0] << " | " << uConf[1][1] << " |\n\n"
          << "| True category | Urgency correct | Total | Accuracy |\n|---|---|---|---|\n";
        for (int c = 0; c < NCAT; ++c) f << "| " << CATS[c].tag << " | " << uOk[c] << " | " << uTot[c] << " | " << pct(uOk[c], uTot[c]) << " |\n";

        // 6 cost ------------------------------------------------------------------
        f << "\n## 6. Cost\n\n| Cost component | Amount | Per 1,000 incidents |\n|---|---|---|\n"
          << "| Compute (" << num(wallMs / 1000.0, 2) << " s at " << money(cfg.computePerHour) << "/hour) | " << money(computeCost) << " | " << money(computeCost * per1k) << " |\n"
          << "| Human review (" << reviewN << " x " << money(cfg.costReview) << ") | " << money(reviewCostTotal) << " | " << money(reviewCostTotal * per1k) << " |\n"
          << "| Wrong-queue misroutes among auto-handled | " << money(catCostTotal) << " | " << money(catCostTotal * per1k) << " |\n"
          << "| Urgency errors among auto-handled (" << missedUrgAuto << " missed x " << money(cfg.costMissedUrgent) << ", " << falseUrgAuto << " false x " << money(cfg.costFalseUrgent) << ") | "
          << money(urgCostTotal) << " | " << money(urgCostTotal * per1k) << " |\n"
          << "| **Total** | **" << money(totalCost) << "** | **" << money(totalCost * per1k) << "** |\n\n"
          << "Model load (" << (llmReady ? num(loadMs / 1000.0) + " s" : "n/a") << ") is one-off and not included. "
          << "All unit costs are assumptions you set on the command line; replace the placeholders with your real figures before comparing engines.\n\n";

        // 7 costly misroutes -----------------------------------------------------
        f << "## 7. Costly misroutes (auto-handled incidents only)\n\n";
        std::vector<size_t> costly;
        for (size_t i = 0; i < N; ++i) if (!res[i].review && res[i].qualityCost > 0) costly.push_back(i);
        std::sort(costly.begin(), costly.end(), [&](size_t a, size_t b) {
            if (res[a].qualityCost != res[b].qualityCost) return res[a].qualityCost > res[b].qualityCost;
            return a < b;
        });
        if (costly.empty()) f << "None.\n\n";
        else {
            std::map<std::pair<int, int>, std::pair<int, double>> pairs;
            for (size_t i : costly) { auto& e = pairs[{res[i].truthCat, res[i].autoCat}]; e.first++; e.second += res[i].qualityCost; }
            std::vector<std::pair<double, std::pair<int, int>>> pv;
            for (auto& kv : pairs) pv.push_back({kv.second.second, kv.first});
            std::sort(pv.rbegin(), pv.rend());
            f << "| True category | Routed as | Incidents | Total cost |\n|---|---|---|---|\n";
            for (size_t k = 0; k < pv.size() && k < 12; ++k)
                f << "| " << catName(pv[k].second.first) << " | " << catName(pv[k].second.second) << " | " << pairs[pv[k].second].first << " | " << money(pv[k].first) << " |\n";
            f << "\nWorst individual cases (showing " << std::min(costly.size(), CAP) << " of " << costly.size() << "):\n\n"
              << "| ID | True | Routed as | Urgency | Cost | Path | Text |\n|---|---|---|---|---|---|---|\n";
            for (size_t k = 0; k < costly.size() && k < CAP; ++k) {
                size_t i = costly[k]; const Result& r = res[i];
                f << "| " << incidents[i].id << " | " << catPrefix(r.truthCat) << " | " << catPrefix(r.autoCat) << " | "
                  << (r.truthUrg == 0 ? "U" : "S") << "->" << (r.autoUrgent ? "U" : "S") << " | " << money(r.qualityCost) << " | " << r.source << " | "
                  << mdEsc(truncUtf8(incidents[i].text, 100)) << " |\n";
            }
            f << "\n";
        }

        // 8 LLM diagnostics ------------------------------------------------------
        f << "## 8. LLM diagnostics (why LLM scores may look wrong)\n\n";
        if (llmCalls == 0) f << "The LLM answered no incidents in this run, so there is nothing to diagnose. Run with `--mode hybrid` or `--mode llm` and a loaded model.\n\n";
        else {
            int sub = 0, llmOk = 0, ruleOk = 0, agree = 0;
            std::vector<int> predCount(NCAT, 0), trueCount(NCAT, 0);
            std::vector<std::vector<int>> lconf(NCAT, std::vector<int>(NCAT, 0));
            double confOk = 0, confBad = 0; int nOk = 0, nBad = 0;
            int binN[4] = {0, 0, 0, 0}, binOk[4] = {0, 0, 0, 0};
            for (size_t i = 0; i < N; ++i) {
                const Result& r = res[i];
                if (r.llmCat < 0 || r.truthCat < 0) continue;
                ++sub; predCount[r.llmCat]++; trueCount[r.truthCat]++; lconf[r.truthCat][r.llmCat]++;
                bool ok = r.llmCat == r.truthCat;
                if (ok) { ++llmOk; confOk += r.llmConf; ++nOk; } else { confBad += r.llmConf; ++nBad; }
                if (r.ruleCat == r.truthCat) ++ruleOk;
                if (r.ruleCat == r.llmCat) ++agree;
                int b = r.llmConf < 0.5f ? 0 : r.llmConf < 0.8f ? 1 : r.llmConf < 0.95f ? 2 : 3;
                binN[b]++; if (ok) binOk[b]++;
            }
            f << "Evaluated on the " << sub << " incidents the LLM answered"
              << (cfg.mode == "hybrid" ? " (in hybrid mode these are only the cases the rules were unsure about, so they are the hardest ones by construction)" : "") << ".\n\n"
              << "| Measure | Value |\n|---|---|\n"
              << "| LLM accuracy on this subset | " << llmOk << " / " << sub << " = " << pct(llmOk, sub) << " |\n"
              << "| Keyword rules' best guess on the same subset | " << ruleOk << " / " << sub << " = " << pct(ruleOk, sub) << " |\n"
              << "| LLM agrees with rules' best guess | " << pct(agree, sub) << " |\n"
              << "| Mean LLM confidence when right / wrong | " << (nOk ? num(confOk / nOk, 3) : "n/a") << " / " << (nBad ? num(confBad / nBad, 3) : "n/a") << " |\n\n"
              << "If the rules' best guess beats the LLM on this subset, the LLM answer is *lowering* quality and you should lean on human review or fix the prompt.\n\n"
              << "### Answer distribution (look for letter bias)\n\n| Category | LLM chose | Truly is |\n|---|---|---|\n";
            for (int c = 0; c < NCAT; ++c) f << "| " << CATS[c].letter << " = " << CATS[c].tag << " | " << predCount[c] << " | " << trueCount[c] << " |\n";
            f << "\n### Confidence calibration\n\n| LLM confidence | Incidents | Accuracy |\n|---|---|---|\n";
            const char* binName[4] = {"< 0.50", "0.50 - 0.80", "0.80 - 0.95", ">= 0.95"};
            for (int b = 0; b < 4; ++b) f << "| " << binName[b] << " | " << binN[b] << " | " << pct(binOk[b], binN[b]) << " |\n";
            f << "\nIf high-confidence answers are often wrong, the probabilities are over-confident and the expected-cost review rule will under-route to humans.\n\n"
              << "### LLM confusion on its subset (rows = true, columns = LLM answer)\n\n| True \\ LLM |";
            for (int c = 0; c < NCAT; ++c) f << ' ' << CATS[c].prefix << " |";
            f << "\n|---|";
            for (int c = 0; c < NCAT; ++c) f << "---|";
            f << "\n";
            for (int t = 0; t < NCAT; ++t) { f << "| " << CATS[t].tag << " |"; for (int p = 0; p < NCAT; ++p) f << ' ' << lconf[t][p] << " |"; f << "\n"; }
            f << "\n";
        }

        // 9 keywords --------------------------------------------------------------
        f << "## 9. Why the rules decided (top matched keywords)\n\n";
        for (int c = 0; c < NCAT; ++c) {
            std::vector<std::pair<int, std::string>> v;
            for (auto& kv : kwCount[c]) v.push_back({kv.second, kv.first});
            std::sort(v.rbegin(), v.rend());
            f << "- **" << CATS[c].tag << "**: ";
            for (size_t i = 0; i < v.size() && i < 8; ++i) f << (i ? ", " : "") << "`" << v[i].second << "` (" << v[i].first << ")";
            if (v.empty()) f << "(no rule-decided incidents)";
            f << "\n";
        }

        // 10 / 11 lists --------------------------------------------------------------
        f << "\n## 10. Raw category errors (engine alone, before human review)\n\n";
        size_t shown = 0, total = 0;
        for (size_t i = 0; i < N; ++i) if (res[i].truthCat >= 0 && res[i].autoCat != res[i].truthCat) ++total;
        if (!total) f << "None.\n";
        else {
            f << "Showing " << std::min(total, CAP) << " of " << total << ".\n\n| ID | True | Engine | Path | Reviewed | Text |\n|---|---|---|---|---|---|\n";
            for (size_t i = 0; i < N && shown < CAP; ++i) {
                const Result& r = res[i];
                if (r.truthCat < 0 || r.autoCat == r.truthCat) continue;
                f << "| " << incidents[i].id << " | " << catPrefix(r.truthCat) << " | " << catPrefix(r.autoCat) << " | " << r.source
                  << " | " << (r.review ? "yes" : "no") << " | " << mdEsc(truncUtf8(incidents[i].text, 100)) << " |\n";
                ++shown;
            }
        }
        f << "\n## 11. Missed urgent incidents that were NOT caught by review\n\n";
        shown = 0; total = 0;
        for (size_t i = 0; i < N; ++i) if (res[i].truthUrg == 0 && !res[i].autoUrgent && !res[i].review) ++total;
        if (!total) f << "None.\n";
        else {
            f << "Showing " << std::min(total, CAP) << " of " << total << ".\n\n| ID | Engine category | Text |\n|---|---|---|\n";
            for (size_t i = 0; i < N && shown < CAP; ++i) {
                if (!(res[i].truthUrg == 0 && !res[i].autoUrgent && !res[i].review)) continue;
                f << "| " << incidents[i].id << " | " << catName(res[i].autoCat) << " | " << mdEsc(truncUtf8(incidents[i].text, 100)) << " |\n";
                ++shown;
            }
        }

        f << "\n## 12. Notes and limitations\n\n"
          << "- True categories come from the `incident_id` prefix; true urgency from the CSV column.\n"
          << "- Reviewed incidents are assumed to be resolved correctly by the human team. Real reviewers make mistakes and take time, so end-to-end numbers are an upper bound.\n"
          << "- Costs use the unit values you pass in. Compute cost is engine wall time times an hourly rate; it does not include model-load time, hardware purchase, or API fees.\n"
          << "- The keyword weights were tuned on the first dataset, so rule accuracy there is optimistic. Compare engines on a set that was not used for tuning.\n"
          << "- The LLM only chooses the category. Urgency of context-dependent categories is always decided by the keyword cues.\n";
    }

    // ==================== console summary ====================
    std::cout << "Classified " << N << " incidents in " << num(wallMs) << " ms (" << num(wallMs / N, 3) << " ms each)\n"
              << "  rules: " << srcN["rules"] << "  llm: " << srcN["llm"] << "  low-conf rules: " << srcN["rules-lowconf"] << "  unclassified: " << srcN["unclassified"] << "\n"
              << "  human review: " << reviewN << " (" << pct(reviewN, N) << ")   auto-handled missed urgent: " << missedUrgAuto
              << "   raw missed urgent: " << missedUrgRaw << "\n"
              << "  raw category accuracy: " << pct(catCorrectRaw, catKnown) << "   auto-handled accuracy: " << pct(autoCatCorrect, autoKnown) << "\n"
              << "  total cost: " << money(totalCost) << " (" << money(totalCost * per1k) << " per 1,000)\n"
              << "Wrote " << cfg.out << "_report.md, " << cfg.out << "_results.csv, " << cfg.out << "_metrics.json\n";
    return 0;
}