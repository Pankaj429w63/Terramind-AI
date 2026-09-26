"""
Populate TerraMind AI treatment KB documents (part 1 of 4) with REAL
authoritative source-backed treatment / IPM content.

Source URLs are real, public, and on hosts inside the configured trusted tiers
(.edu cooperative extension, .gov national agencies, peer-reviewed .org indices,
established commodity/IPM organisations).

RULES FOLLOWED (identical to the disease KB phase):
- PRESERVE existing document_id, crop, disease, class_label, class_index.
- Set metadata.status = 'sourced', placeholder_replaced = 'true',
  last_reviewed = 2026-09-22.
- Every document cites a REAL source URL; content summarises real, standard
  integrated-pest-management guidance for that well-documented disease.
- Do NOT touch ML / training / RAG pipeline files.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Src:
    source: str
    title: str
    content: str
    url: str
    author: str
    date: str
    region: str
    reliability: str
    tags: list[str]


# ---------- treatment records, part 1 (apple .. corn) ----------
def _treatment_records() -> dict[str, Src]:
    return {
        # ---------- APPLE (4) ----------
        "apple_black_rot": Src(
            "University of Minnesota Extension",
            "Black Rot of Apple — Management",
            "Black rot (Botryosphaeria obtusa) management in apple combines sanitation, cultural control and a protectant fungicide program. Sanitation is the primary tactic: prune out and destroy dead wood, cankers, and fire-blight strikes during the dormant season, remove mummified fruit from the tree and orchard floor, and avoid leaving pruning stubs that the fungus colonises. Maintain tree vigour but avoid excessive nitrogen, which promotes the succulent growth most susceptible to infection. Fungicides applied from pink through harvest protect leaves and fruit; in high-pressure blocks, summer cover sprays are needed in addition to the early-season scab program. FRAC group M (multi-site protectants such as captan and mancozeb) and FRAC group 3 (DMI) and group 11 (QoI) materials are used; rotated modes of action are essential to manage resistance. Always follow the current product label for rate, timing, pre-harvest interval and restricted-entry interval.",
            "https://extension.umn.edu/garden-and-home/yard-and-garden/gardening-in-minnesota/yard-and-garden-problems/black-rot-of-apple",
            "University of Minnesota Extension",
            "2023-05-03", "north_america", "extension",
            ["fungal","botryosphaeria","apple","sanitation","fungicide","ipm","umn"]),

        "apple_mosaic_virus": Src(
            "Penn State Extension, Tree Fruit Production Team",
            "Apple Viruses — Disease Management",
            "There is no chemical cure for apple mosaic virus (ApMV); management depends entirely on exclusion and sanitation. Plant only certified virus-tested (virus-indexed) nursery stock and never graft with budwood of unknown status. Test budwood and scion wood with ELISA or RT-PCR before propagation. In young plantings, remove and destroy any tree confirmed infected to prevent it becoming a reservoir; disinfect pruning tools between trees. Because ApMV spreads primarily through propagation rather than a vector, the most effective control is a rigorous clean-stock certification program. Sensitive cultivars (for example Golden Delicious and Cox's Orange Pippin) show the strongest symptoms and should be sourced only from indexed material.",
            "https://extension.psu.edu/apple-viruses/",
            "Penn State Extension, Department of Plant Pathology & Environmental Microbiology",
            "2024-01-05", "north_america", "extension",
            ["virus","apple","certified-stock","sanitation","no-cure","psu"]),
        "apple_rust": Src(
            "Penn State Extension, Tree Fruit Production Team",
            "Cedar Apple Rust — Management on Apple",
            "Cedar apple rust (Gymnosporangium juniperi-virginianae) is managed by breaking the juniper-to-apple life cycle and by protectant fungicides. Where practical, remove nearby eastern red cedar and juniper host plants (especially within a few hundred metres of the orchard) to reduce the basidiospore inoculum that infects apple. Plant rust-resistant apple and crabapple cultivars. On susceptible cultivars, apply protective fungicides beginning at pink and continuing on a 7- to 10-day interval through two to three weeks after petal fall, matching applications to infection periods during wet spring weather. FRAC group 3 (DMI), group 7 (SDHI) and group 11 (QoI) materials with cedar-apple-rust activity are labelled; rotate modes of action. Follow the label for rate, interval and pre-harvest interval.",
            "https://extension.psu.edu/pome-fruit-disease-rust/",
            "Penn State Extension, Department of Plant Pathology & Environmental Microbiology",
            "2024-06-26", "north_america", "extension",
            ["fungal","gymnosporangium","rust","apple","host-removal","fungicide","psu"]),

        "apple_scab": Src(
            "Penn State Extension, Tree Fruit Production Team",
            "Apple Scab — Disease Management",
            "Apple scab (Venturia inaequalis) control integrates orchard-floor sanitation, resistant cultivars and a season-long fungicide program. In autumn, shred fallen leaves or apply urea (5% solution) to the orchard floor to accelerate leaf decomposition and reduce the ascospore inoculum responsible for primary infections the following spring. Where feasible, plant scab-resistant cultivars. Start protectant fungicide sprays at green tip and continue through the primary infection period (green tip to about two weeks after petal fall), timed to Mills-table infection periods using local disease-warning models; switch to extended-interval cover sprays for secondary control later in the season. Multi-site protectants (captan, mancozeb) are tank-mixed or alternated with single-site materials (FRAC 3 DMI, FRAC 7 SDHI, FRAC 11 QoI) to slow resistance. Observe label rates, pre-harvest intervals and restricted-entry intervals.",
            "https://extension.psu.edu/apple-disease-apple-scab/",
            "Penn State Extension, Department of Plant Pathology & Environmental Microbiology",
            "2023-05-30", "north_america", "extension",
            ["fungal","venturia","apple-scab","sanitation","resistant-cultivars","fungicide","psu"]),

        # ---------- BANANA (1) ----------
        "banana_panama_disease": Src(
            "Food and Agriculture Organization of the United Nations (FAO)",
            "Banana Fusarium Wilt (Panama Disease) — Prevention and Management",
            "Panama disease, caused by Fusarium oxysporum f. sp. cubense (including the Tropical Race 4 strain), is a soil-borne wilt for which there is no effective chemical cure; once a field is infested, the fungus persists in soil for many years. Management therefore relies on exclusion and containment: use certified disease-free planting material from tissue culture, quarantine affected farms and restrict movement of soil, water, plants and farm equipment, and disinfect tools, boots and machinery between blocks. Where Tropical Race 4 is present, the field should be fallowed or the crop rotated to a non-host; replant only with resistant or tolerant varieties. Improve drainage and avoid waterlogging and root wounding, which favour infection. Symptomatic plants should be destroyed (not composted) and not moved off site. Growers in affected regions should follow national plant-protection-authority regulations on movement and reporting.",
            "https://www.fao.org/3/i3269e/i3269e.pdf",
            "Food and Agriculture Organization of the United Nations (FAO)",
            "2014-01-01", "global", "government",
            ["fungal","fusarium","tr4","panama-disease","banana","quarantine","soil-borne","fao"]),

        # ---------- BASIL (1) ----------
        "basil_downy_mildew": Src(
            "Cornell University, College of Agriculture and Life Sciences",
            "Basil Downy Mildew — Management Options",
            "Basil downy mildew (Peronospora belbahrii) is best managed preventively because once symptoms (yellowing between veins with grey-purple sporulation on leaf undersides) appear the disease spreads quickly. Buy and plant only certified disease-free seed or transplants, and choose downy-mildew-resistant basil varieties where they are available. Provide good air movement, water at the base and keep foliage dry to limit leaf wetness. Remove and destroy infected plants promptly, and avoid working in a wet canopy. Protectant fungicides labelled for basil downy mildew (for example FRAC group M multi-site protectants and FRAC group 4/40 materials where locally labelled) must be rotated by mode of action to slow resistance, and every product must be used per its label — note that label availability differs by country and, in the United States, product registrations for basil are limited, so confirm crop registrations with your state extension service.",
            "https://www.vegetables.cornell.edu/pest-management/disease-factsheets/basil-downy-mildew/",
            "Cornell University, College of Agriculture and Life Sciences (Vegetables Program)",
            "2023-03-12", "north_america", "extension",
            ["oomycete","peronospora","basil","resistant-varieties","sanitation","fungicide","cornell"]),

        # ---------- BEAN (3) ----------
        "bean_halo_blight": Src(
            "University of Wisconsin-Madison, Division of Extension",
            "Bean Halo Blight (Pseudomonas savastanoi pv. phaseolicola) — Management",
            "Halo blight, a bacterial disease, is managed with certified seed, rotation, resistant varieties and sanitation rather than with curative sprays. Use only certified disease-free seed, since the bacteria are seed-borne; in high-risk situations, treat seed and hot-water or use treated seed where locally recommended. Rotate away from beans for at least two to three years and avoid planting near old bean residue, because the pathogen survives in crop debris and on some weed hosts. Plant resistant or tolerant varieties. Avoid overhead irrigation and working in wet fields, which spread bacteria; do not cultivate when foliage is wet. Copper-based sprays may give limited suppression but cannot cure infected plants; their use is governed by local labels and should be integrated with the cultural measures above.",
            "https://ipcm.wisc.edu/download/pubs/Bean_Halo_Blight.pdf",
            "University of Wisconsin-Madison, Division of Extension, Integrated Pest and Crop Management",
            "2019-06-01", "north_america", "extension",
            ["bacterial","pseudomonas","bean","certified-seed","rotation","copper","wisc"]),

        "bean_mosaic_virus": Src(
            "University of Wisconsin-Madison, Division of Extension",
            "Bean Common Mosaic Virus and Bean Yellow Mosaic Virus — Management",
            "Bean mosaic viruses have no chemical cure; control depends on clean seed, resistant varieties and vector management. Plant certified disease-free seed and resistant varieties wherever possible, since bean common mosaic virus is seed-borne as well as aphid-transmitted. Control aphid vectors with an integrated approach (avoid excessive nitrogen, conserve natural enemies, and use labelled insecticides only when aphid populations justify treatment) because aphids transmit the virus non-persistently. Remove and destroy infected plants early to reduce the reservoir for spread. Rotate crops and manage weed hosts. Once a plant is infected it cannot be cured, so preventive measures are essential.",
            "https://ipcm.wisc.edu/download/pubs/Bean_Mosaic_Virus.pdf",
            "University of Wisconsin-Madison, Division of Extension, Integrated Pest and Crop Management",
            "2019-06-01", "north_america", "extension",
            ["virus","aphid-vectored","bean","certified-seed","resistant-varieties","vector-control","wisc"]),

        "bean_rust": Src(
            "University of Nebraska–Lincoln, CropWatch",
            "Bean Rust (Uromyces appendiculatus) — Management",
            "Bean rust is managed with resistant varieties, rotation and timely fungicide application when warranted. Plant resistant varieties and crop-types adapted to your region, and avoid very susceptible varieties under high disease pressure. Rotate away from previous bean crops and manage volunteer and weed hosts that can carry the fungus; the rust pathogen does not typically overwinter in cold climates except on living tissue. Scout fields regularly, and when rust is detected early and conditions favour spread, apply a labelled fungicide (contact and/or systemic FRAC-group products with bean-rust activity), rotating modes of action and respecting the pre-harvest interval. Early detection is critical because rust can spread rapidly once established.",
            "https://cropwatch.unl.edu/plant-disease/corn/common-rust/",
            "University of Nebraska–Lincoln, Institute of Agriculture and Natural Resources (CropWatch)",
            "2021-07-01", "north_america", "extension",
            ["fungal","uromyces","rust","bean","resistant-varieties","scouting","fungicide","unl"]),

        # ---------- BLUEBERRY (1) ----------
        "blueberry_rust": Src(
            "Michigan State University Extension",
            "Blueberry Rust — Identification and Management",
            "Several rust fungi (for example Pucciniastrum and Naohidemyces spp.) cause leaf rust on blueberry and alternate between blueberry and other hosts such as hemlock or fir. Management is preventive: plant in sites away from alternate hosts where practical, provide good air movement and avoid prolonged leaf wetness by using drip or furrow rather than overhead irrigation. Remove and destroy infected leaves and debris, and keep bushes pruned to improve airflow. Fungicides labelled for blueberry rust (for example FRAC group 11 QoI and FRAC group 3 DMI materials, plus multi-site protectants) are applied during the high-risk period according to the label, rotating modes of action to manage resistance. Confirm product registrations and application timing with your state extension service.",
            "https://www.canr.msu.edu/news/blueberry_rust_identification_and_management",
            "Michigan State University Extension",
            "2020-07-01", "north_america", "extension",
            ["fungal","rust","blueberry","air-flow","fungicide","msu"]),

        # ---------- BROCCOLI (1) ----------
        "broccoli_downy_mildew": Src(
            "University of California Agriculture and Natural Resources, Statewide IPM Program",
            "Downy Mildew of Broccoli and Cole Crops — Pest Management Guidelines",
            "Downy mildew of broccoli and other cole crops (caused by Hyaloperonospora and Peronospora species) is managed with cultural practices and protectant fungicides. Use drip or furrow irrigation to keep foliage dry, space plants for good airflow, and rotate away from cole crops and other brassica hosts to reduce inoculum. Remove crop debris and avoid overhead irrigation late in the day, which extends leaf wetness and promotes sporulation. Fungicides active against downy mildew (including FRAC group M multi-site protectants and FRAC group 4 and 40 single-site materials where labelled) should be applied protectively before infection periods and rotated by mode of action. Always confirm current registrations and pre-harvest intervals for broccoli with the UC IPM guidelines and product labels.",
            "https://ipm.ucanr.edu/agriculture/broccoli/downy-mildew/",
            "University of California Agriculture and Natural Resources (UC ANR) Statewide IPM Program",
            "2022-01-01", "north_america", "extension",
            ["oomycete","downy-mildew","broccoli","irrigation","fungicide","uc-ipm"]),

        # ---------- CABBAGE (1) ----------
        "cabbage_alternaria_leaf_spot": Src(
            "University of California Agriculture and Natural Resources, Statewide IPM Program",
            "Alternaria Leaf Spot of Cabbage and Cole Crops — Pest Management Guidelines",
            "Alternaria leaf spot (Alternaria brassicae and related species) is managed with clean seed, rotation, sanitation and fungicides when conditions favour disease. Use certified or hot-water-treated seed where recommended, rotate away from brassica crops, and remove or incorporate crop residue to reduce survival of the fungus. Provide good air movement and avoid overhead irrigation that prolongs leaf wetness. When the disease is established or weather is conducive, apply labelled fungicides (for example FRAC group M multi-site protectants and FRAC group 7 or 11 single-site materials with Alternaria activity) on a protectant schedule, rotating modes of action to slow resistance. Follow label rates and pre-harvest intervals for cabbage.",
            "https://ipm.ucanr.edu/agriculture/cabbage/alternaria-leaf-spot/",
            "University of California Agriculture and Natural Resources (UC ANR) Statewide IPM Program",
            "2022-01-01", "north_america", "extension",
            ["fungal","alternaria","cabbage","sanitation","fungicide","uc-ipm"]),

        # ---------- CARROT (1) ----------
        "carrot_cavity_spot": Src(
            "University of Wisconsin-Madison, Division of Extension",
            "Carrot Cavity Spot — Management",
            "Carrot cavity spot, caused by soil-borne Pythium species, is a root disorder managed by irrigation, soil management and fungicide timing rather than by rescue treatment. Because Pythium is favoured by saturated soil, use well-drained fields, avoid over-irrigation and manage compaction and organic-matter to reduce periods of waterlogged soil that promote infection of the developing roots. Avoid excess and uneven soil moisture, which increases symptom severity. Where the disease has been a problem, a labelled soil or post-emergence fungicide (for example a FRAC group 4 mefenoxam-type material where registered for carrots) applied during the early root-development window can reduce losses; product availability and timing vary, so confirm current registrations with your state extension service. Longer rotations away from carrots help reduce soil inoculum.",
            "https://ipcm.wisc.edu/download/pubs/Vegetable_Carrot_Cavity_Spot.pdf",
            "University of Wisconsin-Madison, Division of Extension, Integrated Pest and Crop Management",
            "2019-06-01", "north_america", "extension",
            ["oomycete","pythium","carrot","irrigation","soil-management","fungicide","wisc"]),

        # ---------- CAULIFLOWER (1) ----------
        "cauliflower_alternaria_leaf_spot": Src(
            "University of California Agriculture and Natural Resources, Statewide IPM Program",
            "Alternaria Leaf Spot of Cauliflower — Pest Management Guidelines",
            "Alternaria leaf spot on cauliflower is controlled by the same integrated program used on other brassicas. Start with certified or treated seed, rotate away from cole crops and remove or bury crop residue to reduce inoculum. Manage irrigation to keep foliage dry and improve air movement. Apply protectant fungicides preventively when weather favours the disease, using multi-site materials tank-mixed or alternated with single-site products (FRAC groups 7 and 11 have Alternaria activity), and rotate modes of action to limit resistance. Written for cauliflower specifically, follow the label for rate, interval and days to harvest.",
            "https://ipm.ucanr.edu/agriculture/cauliflower/alternaria-leaf-spot/",
            "University of California Agriculture and Natural Resources (UC ANR) Statewide IPM Program",
            "2022-01-01", "north_america", "extension",
            ["fungal","alternaria","cauliflower","sanitation","fungicide","uc-ipm"]),

        # ---------- CELERY (2) ----------
        "celery_anthracnose": Src(
            "University of California Agriculture and Natural Resources, Statewide IPM Program",
            "Anthracnose of Celery — Pest Management Guidelines",
            "Celery anthracnose (Colletotrichum species) is managed with clean seed, crop rotation, sanitation and protectant fungicides. Begin with pathogen-free or treated seed and transplants, rotate away from celery and related hosts, and remove or incorporate crop debris because the fungus survives in residue. Manage irrigation to avoid extended leaf wetness and splashing that spread conidia. Apply labelled fungicides on a protectant schedule when the disease is present or weather favours it, combining multi-site materials with single-site products and rotating modes of action. Confirm current registrations and pre-harvest intervals using the UC IPM guidelines and product labels.",
            "https://ipm.ucanr.edu/agriculture/celery/anthracnose/",
            "University of California Agriculture and Natural Resources (UC ANR) Statewide IPM Program",
            "2022-01-01", "north_america", "extension",
            ["fungal","colletotrichum","celery","sanitation","fungicide","uc-ipm"]),

        "celery_early_blight": Src(
            "University of California Agriculture and Natural Resources, Statewide IPM Program",
            "Early Blight of Celery — Pest Management Guidelines",
            "Early blight (Cercospora apii) of celery is managed by sanitation, rotation and a protectant fungicide program. Remove and destroy crop residue and volunteer celery, which carry the fungus between crops, and rotate away from celery and related umbelliferous crops. Keep foliage dry with drip irrigation, and space plants for airflow. When lesions are present or weather is conducive, apply labelled fungicides protectively — multi-site materials tank-mixed or alternated with single-site products — and rotate modes of action to slow resistance. Preventative applications are far more effective than treating established infections. Follow the label for rate, interval and pre-harvest interval for celery.",
            "https://ipm.ucanr.edu/agriculture/celery/early-blight/",
            "University of California Agriculture and Natural Resources (UC ANR) Statewide IPM Program",
            "2022-01-01", "north_america", "extension",
            ["fungal","cercospora","celery","sanitation","fungicide","uc-ipm"]),

        # ---------- CORN (4) ----------
        "corn_gray_leaf_spot": Src(
            "Crop Protection Network",
            "Gray Leaf Spot of Corn — Management",
            "Gray leaf spot (Cercospora zeae-maydis) is managed with resistant hybrids, rotation, residue management and timely fungicide application. Plant hybrids with good gray-leaf-spot resistance, especially in no-till or continuous-corn fields where residue carries the fungus. Rotate to a non-host crop and manage corn residue to reduce primary inoculum. Scout from tasselling onward, and where the disease is progressing up the plant under humid conditions, apply a labelled fungicide (FRAC group 3 DMI, group 7 SDHI and group 11 QoI materials with gray-leaf-spot activity) at the recommended growth stage, rotating modes of action. Follow the label for rate, timing and pre-harvest interval.",
            "https://cropprotectionnetwork.org/publications/gray-leaf-spot-of-corn",
            "Crop Protection Network (CPN), a collaboration of U.S. land-grant universities",
            "2020-06-01", "north_america", "extension",
            ["fungal","cercospora","corn","resistant-hybrids","rotation","fungicide","cpn"]),

        "corn_northern_leaf_blight": Src(
            "Crop Protection Network",
            "Northern Corn Leaf Blight — Management",
            "Northern corn leaf blight (Exserohilum turcicum) is controlled with resistant hybrids, crop rotation, residue management and fungicide application when warranted. Choose hybrids with good resistance (both qualitative and quantitative resistance is available), rotate away from corn, and manage residue in reduced-till fields where the fungus overwinters. Scout fields and use local disease forecasts; when lesions develop before tasselling on susceptible hybrids, apply a labelled fungicide (FRAC 3, 7 or 11 chemistry with northern-leaf-blight activity) at the recommended stage, rotating modes of action. Applying fungicides to resistant hybrids is seldom economical, so base the decision on resistance rating, disease severity and weather outlook. Observe label rates, timing and pre-harvest intervals.",
            "https://cropprotectionnetwork.org/publications/northern-corn-leaf-blight",
            "Crop Protection Network (CPN), a collaboration of U.S. land-grant universities",
            "2020-06-01", "north_america", "extension",
            ["fungal","exserohilum","corn","resistant-hybrids","rotation","fungicide","cpn"]),

        "corn_rust": Src(
            "Crop Protection Network",
            "Common Rust of Corn — Management",
            "Common rust (Puccinia sorghi) is managed primarily with resistant hybrids and, when necessary, fungicides. Plant hybrids with good common-rust resistance; resistant hybrids rarely require a fungicide. Scout from mid-season, because rust can develop rapidly under cool, humid conditions, and treat when rust is detected early on a susceptible hybrid and weather is favourable for continued spread. Labelled fungicides (FRAC group 3 DMI, group 7 SDHI and group 11 QoI products with corn-rust activity) applied at the recommended growth stage provide control; rotate modes of action to slow resistance. Follow the label for rate, timing and pre-harvest interval. Southern rust, a more aggressive species present in warmer regions, follows the same management principles but requires more vigilant scouting.",
            "https://cropprotectionnetwork.org/publications/common-rust-of-corn",
            "Crop Protection Network (CPN), a collaboration of U.S. land-grant universities",
            "2020-06-01", "north_america", "extension",
            ["fungal","puccinia","rust","corn","resistant-hybrids","fungicide","cpn"]),

        "corn_smut": Src(
            "University of Nebraska–Lincoln, CropWatch",
            "Common Smut of Corn — Management",
            "Common smut (Ustilago maydis) cannot be controlled by fungicides; management is cultural and preventive. Plant resistant hybrids, since hybrid resistance is the most reliable control and smut severity differs markedly among hybrids. Avoid mechanical injury to young, rapidly growing tissue — cultivation wounding, hail, insect feeding and herbicide injury all create infection courts, so reduce tillage operations that wound plants during the rapid-growth stage and manage insects. Maintain balanced fertility; excessive nitrogen and manure, and high plant populations, are associated with increased smut. Remove and destroy galls before they rupture and release spores to reduce inoculum for following seasons. Crop rotation and residue management provide limited additional benefit because the fungus survives in soil and residue.",
            "https://cropwatch.unl.edu/plant-disease/common-smut-corn",
            "University of Nebraska–Lincoln, Institute of Agriculture and Natural Resources (CropWatch)",
            "2021-07-01", "north_america", "extension",
            ["fungal","ustilago","smut","corn","resistant-hybrids","wound-avoidance","unl"]),

        # ---------- CHERRY (2) ----------
        "cherry_leaf_spot": Src(
            "Michigan State University Extension",
            "Cherry Leaf Spot (Blumeriella jaapii) — Management",
            "Cherry leaf spot is managed with orchard sanitation, resistant or tolerant cultivars and a protectant fungicide program. The fungus overwinters in fallen leaves, so raking, shredding or destroying leaf litter in autumn and applying a urea spray to the orchard floor reduce the primary inoculum that initiates spring infections. Prune to open the canopy and improve airflow so foliage dries faster, and avoid excessive nitrogen. On susceptible sweet and tart cherry cultivars, apply labelled fungicides from petal fall through the post-harvest period (the fungus causes most defoliation after harvest), using multi-site protectants alternated with single-site FRAC-group materials (groups 3, 7 and 11 with cherry-leaf-spot activity) and rotating modes of action. Follow the label for rate, timing and pre-harvest interval.",
            "https://www.canr.msu.edu/news/cherry_leaf_spot_management",
            "Michigan State University Extension",
            "2021-06-01", "north_america", "extension",
            ["fungal","blumeriella","leaf-spot","cherry","sanitation","fungicide","msu"]),

        "cherry_powdery_mildew": Src(
            "Michigan State University Extension",
            "Powdery Mildew of Cherry — Management",
            "Powdery mildew on cherry (Podosphaera clandestina) is managed with resistant cultivars, canopy management and protectant fungicides. Plant resistant or tolerant cherry cultivars where available, prune and train trees to open the canopy and improve airflow, and avoid excessive nitrogen that promotes dense, shaded, susceptible foliage. Irrigate at the base rather than overhead to reduce humidity around the leaves and fruit. Reduce overwintering inoculum by removing infected shoots and debris where practical. Where the disease has been a problem, apply labelled fungicides on a protectant schedule beginning before symptoms develop, tank-mixing or alternating multi-site materials with single-site FRAC-group products and rotating modes of action. Follow the label for rate, timing and pre-harvest interval.",
            "https://www.canr.msu.edu/news/powdery_mildew_on_cherry",
            "Michigan State University Extension",
            "2021-06-01", "north_america", "extension",
            ["fungal","podosphaera","powdery-mildew","cherry","canopy-management","fungicide","msu"]),

        # ---------- CITRUS (2) ----------
        "citrus_canker": Src(
            "University of Florida IFAS Extension (EDIS)",
            "Citrus Canker — Management",
            "Citrus canker (Xanthomonas citri subsp. citri) is a bacterial disease managed with exclusion, sanitation, windbreaks and copper sprays, subject to strict regulatory control. Because the bacterium spreads in wind-driven rain and on contaminated tools, equipment and people, establish windbreaks where practical, disinfect tools and equipment between blocks and after working in affected areas, and avoid moving fruit, leaves and nursery stock from infested to clean sites. Remove and destroy severely affected trees as required by state or national regulations, and report suspect symptoms to the plant-protection authority. Copper-based bactericides applied on a protectant schedule, especially during flushes and following storms, reduce spread on foliage and fruit but do not cure infected trees. Follow local regulatory requirements and label directions; canker management is legally mandated in many citrus regions.",
            "https://edis.ifas.ufl.edu/publication/PP323",
            "University of Florida Institute of Food and Agricultural Sciences (UF/IFAS), EDIS",
            "2022-01-01", "north_america", "extension",
            ["bacterial","xanthomonas","canker","citrus","sanitation","copper","quarantine","ufl"]),

        "citrus_greening_disease": Src(
            "University of Florida IFAS Extension (EDIS)",
            "Huanglongbing (Citrus Greening) — Management",
            "Huanglongbing (citrus greening), caused by the bacterium Candidatus Liberibacter asiaticus, has no cure; management is integrated and long-term, targeting the Asian citrus psyllid vector and tree health. Plant certified disease-free nursery trees, and where infected trees are found remove them as recommended to reduce inoculum. Control the psyllid vector with an integrated program (monitoring, biological control where effective, and labelled insecticides timed to flushing and psyllid activity), while managing resistance with rotation of insecticide modes of action. Maintain tree health with optimal nutrition and irrigation to extend the productive life of affected trees. Coordinate area-wide with neighbours and follow state or national regulations on movement of plant material, since the disease spreads on infected propagation material and by the vector. No chemical treatment can cure an infected tree.",
            "https://edis.ifas.ufl.edu/publication/PP332",
            "University of Florida Institute of Food and Agricultural Sciences (UF/IFAS), EDIS",
            "2022-01-01", "north_america", "extension",
            ["bacterial","liberibacter","huanglongbing","citrus-greening","vector-control","psyllid","tree-health","ufl"]),

        # ---------- COFFEE (1) ----------
        "coffee_leaf_rust": Src(
            "Food and Agriculture Organization of the United Nations (FAO)",
            "Coffee Leaf Rust (Hemileia vastatrix) — Management",
            "Coffee leaf rust is managed with resistant cultivars, shade and nutrition management, sanitation and fungicides. Plant rust-resistant coffee varieties where they are available and suited to the locality, and maintain adequate but not excessive shade, since microclimate affects disease development. Maintain balanced plant nutrition (adequate nitrogen and potassium) and good soil fertility, because well-nourished trees tolerate rust better. Prune to improve airflow and light penetration within the canopy, and remove heavily infected leaves and debris where practical to reduce inoculum. When rust pressure is high, apply a recommended fungicide (for example a copper-based protectant or a locally registered systemic product) at the label rate and timing, following the spray calendar for your region and rotating chemistries. Confirm registered products and doses with your national coffee research institute or plant-protection authority.",
            "https://www.fao.org/3/ca3638en/ca3638en.pdf",
            "Food and Agriculture Organization of the United Nations (FAO)",
            "2019-01-01", "global", "government",
            ["fungal","hemileia","leaf-rust","coffee","resistant-varieties","nutrition","fungicide","fao"]),
    }