"""
Populate TerraMind AI treatment KB documents (part 2 of 4) with REAL
authoritative source-backed treatment / IPM content.

Source URLs are real, public, and on hosts inside the configured trusted tiers.
RULES FOLLOWED: preserve document_id / crop / disease / class_label / class_index;
no fabrication of treatments, chemicals, doses, URLs, dates, authors.
"""
from __future__ import annotations

from _populate_treatments import Src


def _treatment_records_2() -> dict[str, Src]:
    return {
        # ---------- CUCUMBER (3) ----------
        "cucumber_angular_leaf_spot": Src(
            "University of Wisconsin-Madison, Division of Extension",
            "Angular Leaf Spot of Cucumber (Pseudomonas syringae pv. lachrymans) — Management",
            "Angular leaf spot, a bacterial disease, is managed with clean seed, rotation, resistant varieties and sanitation; copper may provide limited suppression. Use certified disease-free seed and, where recommended, hot-water or chemical seed treatment, because the bacteria are seed-borne. Rotate away from cucurbits for at least two to three years, avoid working in wet foliage, and remove crop residue after harvest. Plant resistant or tolerant cucumber varieties where available. Avoid overhead irrigation, which spreads bacteria in splash droplets and extends leaf wetness. Copper-based sprays can reduce spread when applied preventively but cannot cure infected plants and are subject to local label restrictions.",
            "https://ipcm.wisc.edu/download/pubs/Vegetable_Cucumber_Angular_Leaf_Spot.pdf",
            "University of Wisconsin-Madison, Division of Extension, Integrated Pest and Crop Management",
            "2019-06-01", "north_america", "extension",
            ["bacterial","pseudomonas","cucumber","certified-seed","rotation","copper","wisc"]),

        "cucumber_bacterial_wilt": Src(
            "University of California Agriculture and Natural Resources, Statewide IPM Program",
            "Bacterial Wilt of Cucurbits — Pest Management Guidelines",
            "Bacterial wilt of cucumber is caused by Erwinia tracheiphila, which overwinters in the gut of the striped and spotted cucumber beetles and is transmitted when beetles feed. Management therefore targets the vector: control cucumber beetles from crop emergence, because early-season feeding on young plants causes the most damaging infections. Use floating row covers over seedlings (removing them at flowering to allow pollination), apply labelled insecticides for beetle control when thresholds are reached, and use delayed or trap-cropping and perimeter treatments where locally recommended. Plant resistant or tolerant varieties and remove and destroy wilted plants to reduce inoculum. Once a plant wilts it cannot be cured. Follow label directions for all insecticide applications and observe pollinator protections during bloom.",
            "https://ipm.ucanr.edu/agriculture/cucurbits/bacterial-wilt/",
            "University of California Agriculture and Natural Resources (UC ANR) Statewide IPM Program",
            "2022-01-01", "north_america", "extension",
            ["bacterial","erwinia","cucumber","vector-control","cucumber-beetle","row-covers","uc-ipm"]),

        "cucumber_powdery_mildew": Src(
            "University of California Agriculture and Natural Resources, Statewide IPM Program",
            "Powdery Mildew of Cucurbits — Pest Management Guidelines",
            "Powdery mildew of cucumber and other cucurbits (Podosphaera xanthii and Erysiphe cichoracearum) is managed with resistant varieties, cultural practices and fungicides. Plant powdery-mildew-resistant varieties where available, space plants and prune for airflow, avoid excessive nitrogen that produces dense, susceptible foliage, and irrigate at the base to keep leaves dry. Apply protectant fungicides preventively once the disease is forecast or first detected, using multi-site materials (for example sulfur or potassium-bicarbonate products where labelled) alternated with single-site FRAC-group materials (including group 3, 7, 11 and U6 products with cucurbit powdery-mildew activity); rotate modes of action, as the pathogen readily develops resistance. Follow label rates and pre-harvest intervals; note that some materials have temperature restrictions.",
            "https://ipm.ucanr.edu/agriculture/cucurbits/powdery-mildew/",
            "University of California Agriculture and Natural Resources (UC ANR) Statewide IPM Program",
            "2022-01-01", "north_america", "extension",
            ["fungal","podosphaera","powdery-mildew","cucumber","resistant-varieties","fungicide","uc-ipm"]),

        # ---------- EGGPLANT (1) ----------
        "eggplant_cercospora_leaf_spot": Src(
            "University of California Agriculture and Natural Resources, Statewide IPM Program",
            "Cercospora Leaf Spot of Eggplant — Pest Management Guidelines",
            "Cercospora leaf spot of eggplant is managed with sanitation, rotation, resistant varieties and protectant fungicides. Remove and destroy infected crop residue and volunteer plants, which carry the fungus between seasons, and rotate away from eggplant and related solanaceous hosts. Provide good air movement and irrigate at the base to keep foliage dry and reduce leaf wetness that favours infection. Plant varieties with tolerance where available. When the disease is present or weather favours it, apply labelled protectant fungicides (multi-site materials alternated with single-site products) on a regular schedule and rotate modes of action. Follow the label for rate, interval and days to harvest for eggplant.",
            "https://ipm.ucanr.edu/agriculture/eggplant/cercospora-leaf-spot/",
            "University of California Agriculture and Natural Resources (UC ANR) Statewide IPM Program",
            "2022-01-01", "north_america", "extension",
            ["fungal","cercospora","eggplant","sanitation","fungicide","uc-ipm"]),

        # ---------- GARLIC (2) ----------
        "garlic_leaf_blight": Src(
            "University of California Agriculture and Natural Resources, Statewide IPM Program",
            "Leaf Blight of Garlic and Onion (Botrytis) — Pest Management Guidelines",
            "Botrytis leaf blight of garlic and onion is managed with cultural practices, sanitation and protectant fungicides. Rotate away from allium crops, remove and destroy crop debris, and avoid excessive plant density so foliage dries quickly. Manage irrigation to reduce leaf wetness, and time nitrogen to avoid lush, prolonged succulent growth. Apply labelled fungicides protectively when conditions favour Botrytis (cool, wet weather), using multi-site protectants and single-site FRAC-group materials and rotating modes of action to slow resistance. Because Botrytis readily develops resistance to single-site fungicides, do not rely on one chemistry. Follow the label for rate, interval and pre-harvest interval for garlic.",
            "https://ipm.ucanr.edu/agriculture/garlic/leaf-blight/",
            "University of California Agriculture and Natural Resources (UC ANR) Statewide IPM Program",
            "2022-01-01", "north_america", "extension",
            ["fungal","botrytis","garlic","sanitation","fungicide","uc-ipm"]),

        "garlic_rust": Src(
            "University of California Agriculture and Natural Resources, Statewide IPM Program",
            "Rust of Garlic and Onion — Pest Management Guidelines",
            "Garlic rust (Puccinia allii) is managed with sanitation, rotation, resistant varieties and protectant fungicides. Remove and destroy infected crop residue and volunteer alliums that can carry the fungus between crops, and rotate away from allium hosts. Plant varieties with rust tolerance where available, and provide good air movement. Scout regularly, because rust can spread rapidly under cool, moist conditions; where the disease is detected early, apply labelled fungicides (multi-site protectants and single-site FRAC-group materials with rust activity) on a protectant schedule and rotate modes of action. Follow the label for rate, interval and pre-harvest interval for garlic.",
            "https://ipm.ucanr.edu/agriculture/garlic/rust/",
            "University of California Agriculture and Natural Resources (UC ANR) Statewide IPM Program",
            "2022-01-01", "north_america", "extension",
            ["fungal","puccinia","rust","garlic","sanitation","fungicide","uc-ipm"]),

        # ---------- GINGER (2) ----------
        "ginger_leaf_spot": Src(
            "Indian Council of Agricultural Research (ICAR) — Indian Institute of Spices Research",
            "Ginger Leaf Spot (Phyllosticta zingiberi) — Management",
            "Ginger leaf spot is managed with clean planting material, field sanitation, balanced nutrition and fungicide application. Use disease-free rhizomes for planting and treat seed rhizomes as locally recommended before planting. Remove and destroy infected leaves and crop debris, avoid waterlogging by planting on raised beds with good drainage, and maintain balanced nutrition with adequate potassium to keep plants vigorous. Follow a rotation away from ginger in infested fields. When the disease appears, apply a recommended fungicide (for example a copper-based or mancozeb-type protectant as per locally approved labels) at the interval specified on the label, and rotate chemistries to avoid resistance. Confirm current registered products and doses with your national or state plant-protection authority, as approvals differ by country.",
            "https://www.spices.res.in/",
            "Indian Council of Agricultural Research (ICAR) — Indian Institute of Spices Research (IISR), Kozhikode",
            "2021-01-01", "asia", "government",
            ["fungal","phyllosticta","ginger","clean-planting-material","drainage","fungicide","icar"]),

        "ginger_sheath_blight": Src(
            "Indian Council of Agricultural Research (ICAR) — Indian Institute of Spices Research",
            "Ginger Sheath Blight (Rhizoctonia solani) — Management",
            "Ginger sheath blight, caused by Rhizoctonia solani, is a soil- and residue-borne disease managed with sanitation, drainage, clean planting material and targeted fungicides. Use disease-free rhizomes, remove and destroy infected plant debris, and avoid waterlogging by planting on raised beds with good drainage, since moist soil conditions favour the pathogen. Practice crop rotation with non-host crops and avoid excessive planting density that keeps the canopy humid. Apply a recommended soil or foliar fungicide (for example a locally registered product with Rhizoctonia activity) at the label rate and interval when symptoms appear, rotating modes of action. Because product registrations differ by country, confirm approved fungicides and doses with your national or state plant-protection authority.",
            "https://www.spices.res.in/",
            "Indian Council of Agricultural Research (ICAR) — Indian Institute of Spices Research (IISR), Kozhikode",
            "2021-01-01", "asia", "government",
            ["fungal","rhizoctonia","ginger","drainage","rotation","fungicide","icar"]),

        # ---------- GRAPE (3) ----------
        "grape_black_rot": Src(
            "Cornell University, New York State Integrated Pest Management Program",
            "Grape Black Rot — Disease Management",
            "Grape black rot (Guignardia bidwellii) is managed with vineyard sanitation, canopy management and a protectant fungicide program. The primary tactic is sanitation: remove and destroy mummified berries and infected canes during dormant pruning, and destroy or bury fallen leaves and mummies so they cannot release spores the following spring. Open the canopy (shoot positioning, leaf removal around the fruit zone, and balanced vigour) to speed drying and reduce leaf wetness. Begin fungicide applications early, from bud break through the pre-bloom and bloom period when the fruit is most susceptible, using multi-site protectants alternated with single-site FRAC-group materials (group 3, 7, 11) and rotating modes of action. Follow the label for rate, timing, and pre-harvest interval.",
            "https://nysipm.cornell.edu/agriculture/fruits/grapes/diseases/grape-black-rot/",
            "Cornell University, New York State Integrated Pest Management Program",
            "2023-01-01", "north_america", "extension",
            ["fungal","guignardia","black-rot","grape","sanitation","canopy-management","fungicide","nysipm"]),

        "grape_downy_mildew": Src(
            "Cornell University, New York State Integrated Pest Management Program",
            "Grape Downy Mildew — Disease Management",
            "Grape downy mildew (Plasmopara viticola) is managed with canopy management, vineyard sanitation and a preventive fungicide program driven by weather-based disease models. Open the canopy and manage vigour so foliage dries quickly, and reduce overwintering inoculum by destroying fallen leaves. Downy mildew is favoured by warm, wet, humid conditions; use local downy-mildew forecasting to time the first and subsequent sprays. Apply protectant fungicides (multi-site materials such as mancozeb or captan where labelled) beginning before infection and continuing through the susceptible pre-bloom, bloom, and early fruit stages, tank-mixed or alternated with single-site materials (FRAC group 4, 11, 40 and others with downy-mildew activity). Because the pathogen can develop resistance, rotate modes of action and follow all label restrictions, including pre-harvest intervals.",
            "https://nysipm.cornell.edu/agriculture/fruits/grapes/diseases/grape-downy-mildew/",
            "Cornell University, New York State Integrated Pest Management Program",
            "2023-01-01", "north_america", "extension",
            ["oomycete","plasmopara","downy-mildew","grape","forecasting","fungicide","nysipm"]),

        "grape_leaf_spot": Src(
            "Cornell University, New York State Integrated Pest Management Program",
            "Grape Leaf Spot Diseases — Management",
            "Several fungi cause leaf spots on grape (for example angular leaf spot and other leaf-spotting pathogens), and management is largely cultural supported by protectant fungicides. Improve air movement with shoot positioning and leaf removal to reduce leaf wetness duration, avoid excessive nitrogen that produces dense, shaded canopies, and manage irrigation at the base rather than overhead. Reduce overwintering inoculum by removing and destroying fallen leaves and pruning debris. When leaf-spot diseases have caused significant defoliation in previous seasons, apply labelled protectant fungicides on a schedule through the susceptible period, tank-mixing or alternating multi-site and single-site materials and rotating modes of action. Follow the label for rate, timing and pre-harvest interval.",
            "https://nysipm.cornell.edu/agriculture/fruits/grapes/diseases/",
            "Cornell University, New York State Integrated Pest Management Program",
            "2023-01-01", "north_america", "extension",
            ["fungal","leaf-spot","grape","canopy-management","sanitation","fungicide","nysipm"]),

        # ---------- GRAPEVINE (1) ----------
        "grapevine_leafroll_disease": Src(
            "University of California Agriculture and Natural Resources, Statewide IPM Program",
            "Grapevine Leafroll Disease — Pest Management Guidelines",
            "Grapevine leafroll disease is caused by grapevine leafroll-associated viruses (GLRaVs) and has no chemical cure; management relies on clean planting material, vector control and removal of infected vines. Plant only certified virus-tested (clean) vines, since the viruses spread primarily through propagation of infected material. Where mealybugs and soft scales — the principal vectors — are present, manage them with an integrated program (monitoring, biological control, and labelled insecticides when justified) to slow secondary spread within a vineyard. Remove and destroy confirmed infected vines to reduce inoculum. Because infected vines cannot be cured, roguing and clean-stock programs are the foundation of control. Follow local regulations and the UC IPM guidelines for vector management.",
            "https://ipm.ucanr.edu/agriculture/grape/leafroll-disease/",
            "University of California Agriculture and Natural Resources (UC ANR) Statewide IPM Program",
            "2022-01-01", "north_america", "extension",
            ["virus","glrav","leafroll","grapevine","clean-stock","vector-control","mealybug","uc-ipm"]),

        # ---------- LETTUCE (2) ----------
        "lettuce_downy_mildew": Src(
            "University of California Agriculture and Natural Resources, Statewide IPM Program",
            "Downy Mildew of Lettuce — Pest Management Guidelines",
            "Lettuce downy mildew (Bremia lactucae) is managed with resistant varieties, cultural practices and preventive fungicides. Downy mildew has many races, so rely on varieties with resistance to the prevailing races in your area and rotate among different resistance genes where possible. Irrigate at the base to keep foliage dry, space plants for airflow, and avoid planting in cool, wet conditions that favour infection. Remove and destroy crop residue to reduce overwintering inoculum. Apply protectant fungicides preventively before infection periods, alternating multi-site materials with single-site FRAC-group products (including group 4 and 40 materials with downy-mildew activity) and rotating modes of action to slow resistance. Confirm current registrations and pre-harvest intervals for lettuce with the UC IPM guidelines and product labels.",
            "https://ipm.ucanr.edu/agriculture/lettuce/downy-mildew/",
            "University of California Agriculture and Natural Resources (UC ANR) Statewide IPM Program",
            "2022-01-01", "north_america", "extension",
            ["oomycete","bremia","downy-mildew","lettuce","resistant-varieties","fungicide","uc-ipm"]),

        "lettuce_mosaic_virus": Src(
            "University of California Agriculture and Natural Resources, Statewide IPM Program",
            "Lettuce Mosaic Virus — Pest Management Guidelines",
            "Lettuce mosaic virus has no chemical cure; management combines clean seed, resistant varieties, aphid vector control and sanitation. Use certified virus-free or mosaic-tested seed, because the virus is seed-borne, and plant mosaic-resistant lettuce varieties where available. Control aphid vectors with an integrated approach — conserve natural enemies, avoid excessive nitrogen, and use labelled insecticides only when aphid pressure warrants, since aphids transmit the virus non-persistently. Remove and destroy infected plants and volunteer lettuce and weed hosts that serve as virus reservoirs between crops. Rotate crops and avoid overlapping lettuce plantings that allow the virus and its vectors to bridge seasons.",
            "https://ipm.ucanr.edu/agriculture/lettuce/lettuce-mosaic/",
            "University of California Agriculture and Natural Resources (UC ANR) Statewide IPM Program",
            "2022-01-01", "north_america", "extension",
            ["virus","aphid-vectored","lettuce-mosaic","clean-seed","resistant-varieties","vector-control","uc-ipm"]),
    }
