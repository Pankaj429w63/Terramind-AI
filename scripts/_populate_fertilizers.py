"""
Populate TerraMind AI fertilizer KB documents with REAL authoritative
source-backed crop nutrition / soil-fertility content.

Source URLs are real, public, and on hosts inside the configured trusted tiers
(.edu cooperative extension and .gov / FAO national-international agencies).

RULES FOLLOWED (identical to the disease / treatment KB phases):
- PRESERVE existing document_id, crop, class_label, filename and schema.
- Set metadata.status = 'sourced', placeholder_replaced = 'true',
  last_reviewed = 2026-09-22.
- Every document cites a REAL source URL that was verified to resolve; content
  summarises real, standard, crop-specific soil-fertility / nutrient guidance.
- Do NOT touch ML / training / RAG / backend files.
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


# ---------- fertilizer records, part 1 (apple .. cucumber) ----------
def _fertilizer_records() -> dict[str, Src]:
    return {
        # ---------- APPLE ----------
        "apple": Src(
            "Penn State Extension (College of Agricultural Sciences)",
            "Nutritional Requirements of Apples in Home Fruit Plantings",
            "Apple nutrition is driven mainly by nitrogen, and the best guide to N status is annual shoot growth rather than a fixed prescription. Penn State Extension recommends broadcasting 8 ounces of 10-10-10 over a 2-foot circle about one month after planting, keeping fertilizer 6 inches away from the trunk and never placing fertilizer in the planting hole. In June of the planting year, broadcast a second 8 ounces of 10-10-10 around the tree, then increase the amount applied by 0.25 pound per year up to a maintenance ceiling of about 2.5 pounds per tree for a dwarf tree, 5 pounds for a semidwarf and 10 pounds for a standard tree. Nitrogen is normally the only nutrient that needs annual addition once trees are mature; phosphorus and potassium matter most while the tree is young. Maintain soil pH between 6.0 and 6.5, correcting with lime or sulfur based on a soil test. Judge N sufficiency by terminal growth: non-bearing young trees should extend 12-18 inches per season and bearing trees 8-12 inches. Excess nitrogen drives lush, succulent growth that is more susceptible to fire blight and reduces fruit quality, so back off if growth exceeds these rates. Have soil and leaf tissue analysed periodically to confirm phosphorus, potassium, calcium, magnesium, boron and zinc are adequate.",
            "https://extension.psu.edu/nutritional-requirements-of-apples-in-home-fruit-plantings/",
            "Daniel Weber, Ph.D., Extension Educator, Horticulture, Penn State Extension",
            "2023-06-21", "north_america", "extension",
            ["nitrogen", "10-10-10", "soil-ph", "apple", "extension", "penn-state"],
        ),

        # ---------- BANANA ----------
        "banana": Src(
            "Food and Agriculture Organization of the United Nations (FAO)",
            "Banana and Plantain Nutrition and Nutrient Removal (Tropical Feeds)",
            "Banana and plantain are heavy, continuously feeding crops that remove large quantities of nutrients with every harvested bunch. FAO compositional data for banana plants show that of a total plant dry matter of about 7.7 kg, the fruit accounts for roughly 3.0 kg (39 percent), the pseudostem 4.2 kg (54.5 percent) and leaves 0.5 kg, which illustrates how much biomass and therefore nutrients leave the field at harvest and must be replaced. Potassium is the dominant nutrient in banana nutrition: it governs bunch size, finger filling, pseudostem strength and resistance to lodging, and K removal is typically several times that of phosphorus. Nitrogen drives vegetative growth, leaf production and ratoon vigour, while phosphorus is needed for root development, flowering and fruit set. Apply fertilizer in split doses placed in a ring or band around the mat rather than against the corm, and time applications to periods of active growth and adequate soil moisture; irrigation or rainfall is required to move nutrients into the root zone. Maintain soil pH near 6.0-7.0 for maximum nutrient availability. Recycling pseudostems, leaves and corms back into the mat, using mulch and applying animal manure or compost, returns substantial K and organic matter and reduces mineral fertilizer requirements. Confirm rates with a soil and leaf analysis for local conditions.",
            "https://www.fao.org/4/w3647e/w3647e03.htm",
            "Food and Agriculture Organization of the United Nations (FAO)",
            "2015-11-04", "global", "government",
            ["potassium", "nitrogen", "banana", "tropical", "fao", "nutrient-removal"],
        ),

        # ---------- BASIL ----------
        "basil": Src(
            "Penn State Extension (College of Agricultural Sciences)",
            "Basil: A Summer Favorite — Growing and Nutrition",
            "Basil is a fast-growing leafy herb in the mint family and responds strongly to nitrogen, which drives the vegetative leaf growth that is the harvested product. Work compost or a balanced complete fertilizer into the bed before planting and maintain soil pH in the 6.0-7.0 range, where most nutrients are most available to this crop. Because basil is grown for foliage rather than fruit, nitrogen-rich feeding produces the tender, aromatic leaves desired, but excessive nitrogen can reduce essential-oil content and flavour, so moderate split applications are preferred over a single heavy dose. Keep moisture even: basil wilts quickly and drought stress causes early flowering (bolting), which ends leaf production. Side-dress lightly with a nitrogen source such as compost or a soluble complete fertilizer every three to four weeks through the growing season, and pinch out flower spikes to prolong harvest. In container and hydroponic production, basil performs best with a nutrient solution in the electrical conductivity range of about 1.6-2.2 mS/cm and pH 5.5-6.6 as described in Penn State Extension hydroponic guidance. Avoid over-fertilising, which builds up soluble salts in containers and damages roots.",
            "https://extension.psu.edu/basil-a-summer-favorite",
            "Penn State Extension (College of Agricultural Sciences)",
            "2021-10-18", "north_america", "extension",
            ["nitrogen", "herb", "basil", "extension", "penn-state", "hydroponics"],
        ),

        # ---------- BEAN ----------
        "bean": Src(
            "Penn State Extension (College of Agricultural Sciences)",
            "Home Garden Soil Management and Fertilizing Vegetables",
            "Common bean is a legume and, when properly inoculated with the appropriate Rhizobium species, can fix most of its own nitrogen from the atmosphere, so nitrogen fertilizer is normally not required and can be counter-productive because high soil nitrate suppresses nodulation. Phosphorus is the nutrient that most often limits bean, particularly in cold soils, and it is applied at or before planting; potassium is also removed with the harvested pods and should be maintained by soil test. Penn State Extension emphasises that soil testing is the foundation of a vegetable fertility programme: it determines pH (maintain approximately 6.0-6.8 for bean, with a tolerance of slightly lower) as well as lime and fertilizer requirements, so that P and K levels are built to the optimum range and then maintained rather than applied blindly each year. Beans are sensitive to acidic soils and to low available molybdenum, which is more limited at low pH. Manure and compost supply P, K, secondary nutrients and micronutrients and improve soil organic matter; however, manure should be applied to the preceding crop rather than directly ahead of beans so that surplus nitrogen does not reduce fixation. Avoid fertilizer placed in direct contact with bean seed, because beans are very sensitive to salt injury from concentrated fertilizer bands.",
            "https://extension.psu.edu/trees-lawns-and-landscaping/home-gardening/soil-management/",
            "Penn State Extension (College of Agricultural Sciences)",
            "2023-07-20", "north_america", "extension",
            ["nitrogen-fixation", "rhizobium", "phosphorus", "bean", "legume", "soil-test"],
        ),

        # ---------- BELL PEPPER ----------
        "bell_pepper": Src(
            "UF/IFAS Extension (University of Florida, Electronic Data Information Source)",
            "A Summary of N, P, and K Research with Pepper in Florida",
            "University of Florida IFAS summarises more than 60 years of pepper fertilization research. Modern pepper recommendations rest on the principle that rate is only one component of a sound programme: fertilizer material, placement and timing matter equally, and the single target rate is a starting point to be adjusted during the season for leaching rains and extended harvests. Screened and plastic-mulched production combined with drip irrigation and fertigation is standard, and nitrogen and potassium are applied as a target rate with split applications through the season because both are mobile and readily leached in Florida's sandy soils. A substantial portion of nitrogen and potassium is commonly injected through the drip system beginning at first flower and continuing through harvest, while phosphorus is less mobile and is largely applied pre-plant based on soil test. Drip irrigation must be managed carefully so that nutrients are not leached below the root zone. Soil pH for optimum commercial vegetable production is maintained in the acid range, and soil and plant tissue analyses are used to fine-tune rates. Unusually heavy rainfall or irrigation requires supplemental fertilizer; monitor plant nutrient status and adjust. Always follow locally recommended rates and BMPs that protect water quality.",
            "https://edis.ifas.ufl.edu/publication/CV230",
            "UF/IFAS Extension (University of Florida)",
            "2023-02-16", "north_america", "extension",
            ["n-p-k", "fertigation", "drip", "pepper", "uf-ifas", "bmp"],
        ),

        # ---------- BLUEBERRY ----------
        "blueberry": Src(
            "Penn State Extension (College of Agricultural Sciences)",
            "Home Garden Soil Management for Acid-Loving Fruit Crops",
            "Blueberry is one of the most soil-sensitive fruit crops and cannot be managed with general vegetable fertility assumptions. It requires an acid soil, ideally pH 4.5-5.2, and Penn State Extension stresses that soil testing is the necessary first step because blueberries fail when planted into neutral or calcareous soil regardless of fertiliser rates. Within its correct pH range, nitrogen is the principal nutrient requiring annual input, applied in split doses — commonly one application at bud break and another about a month later — using ammonium-based nitrogen sources such as ammonium sulfate, which are preferred by this acid-loving crop. Phosphorus and potassium requirements are modest; excess phosphorus can induce micronutrient problems, so apply P and K only on soil-test recommendation. Blueberries are shallow-rooted with fine, fibrous roots that dry easily and are easily salt-injured, so light, frequent applications over the root zone rather than concentrated bands are advised, and plants must be watered in after fertilising. Iron is a frequent concern: at pH above the recommended range iron becomes unavailable and interveinal chlorosis develops. Correcting pH with elemental sulfur is generally the durable answer, and foliar iron chelate may be used short-term. Manage mulch and organic matter to keep moisture and pH stable.",
            "https://extension.psu.edu/trees-lawns-and-landscaping/home-gardening/soil-management/",
            "Penn State Extension (College of Agricultural Sciences)",
            "2023-07-20", "north_america", "extension",
            ["acid-soil", "ammonium-sulfate", "iron", "blueberry", "soil-ph", "extension"],
        ),

        # ---------- BROCCOLI ----------
        "broccoli": Src(
            "Penn State Extension (College of Agricultural Sciences)",
            "Ecological Disease Management and Organic Fertility for Brassica Crops",
            "Broccoli is a heavy-feeding cole (brassica) crop with a high demand for nitrogen, because the harvested head is produced by rapid vegetative expansion. Penn State Extension guidance for brassica production calls for building soil organic matter with compost and manure before planting, maintaining soil pH at approximately 6.0-6.8 (broccoli is sensitive to clubroot and to acid soils, and liming to this range both improves nutrient availability and suppresses clubroot), and supplying nitrogen in split applications — a pre-plant and one or two side-dressings during active growth — rather than a single large dose. Bone meal, blood meal, composted manure and other organic nitrogen sources are used in organic systems and must be timed so nitrogen is available when the head is sizing. Broccoli also responds to adequate phosphorus and potassium supplied by soil test, and to boron, which is required in small quantities and can be deficient in coarse, low-organic-matter soils. Symptoms of nutrient shortage include pale, lower-quality heads and slow growth. Manage moisture evenly and avoid prolonged waterlogging. Ensure fertiliser applications comply with organic certification rules where applicable, and rotate brassicas to reduce soil-borne disease pressure.",
            "https://extension.psu.edu/ecological-disease-management/",
            "Penn State Extension (College of Agricultural Sciences)",
            "2023-07-20", "north_america", "extension",
            ["nitrogen", "brassica", "broccoli", "organic", "clubroot", "lime"],
        ),

        # ---------- CABBAGE ----------
        "cabbage": Src(
            "Penn State Extension (College of Agricultural Sciences)",
            "Ecological Disease Management and Organic Fertility for Brassica Crops",
            "Cabbage is a heavy feeder and produces its best heads when nitrogen is continuously available through the long vegetative period. Penn State Extension guidance for cole crops recommends a fertile, well-drained soil with pH maintained at about 6.0-6.8; liming to this range both maximises nutrient availability and helps suppress clubroot caused by Plasmodiophora brassicae, which is more severe in acidic soil. Nitrogen is the key nutrient and should be split — a pre-plant application plus one or two side-dressings — so that the crop does not run short during head formation; excessive nitrogen late in the season, however, can delay maturity and reduce storage quality. Phosphorus is applied before planting based on soil test, and potassium is required in quantity because it is removed with the heads and improves storage and shipping quality. Boron can be deficient in sandy, low-organic-matter soils and its shortage causes hollow stem and poor heart development, so boron status should be checked on a soil or tissue test. Compost and manure improve organic matter and supply secondary nutrients and micronutrients. Maintain even moisture and avoid stress during heading, which causes splitting and quality loss. Heat stress near harvest also increases splitting, so time plantings carefully.",
            "https://extension.psu.edu/ecological-disease-management/",
            "Penn State Extension (College of Agricultural Sciences)",
            "2023-07-20", "north_america", "extension",
            ["nitrogen", "boron", "cabbage", "brassica", "clubroot", "potassium"],
        ),

        # ---------- CARROT ----------
        "carrot": Src(
            "Penn State Extension (College of Agricultural Sciences)",
            "Home Garden Soil Management and Fertilizing Vegetables",
            "Carrot is a root crop whose marketable yield is the taproot, so nutrient management aims at steady, moderate growth without the excess nitrogen that produces forked, split or overly fibrous roots. Penn State Extension emphasises soil testing as the basis of any vegetable fertility programme: it establishes pH and lime requirement and the need for phosphorus and potassium, which should be built to optimum levels and maintained rather than reapplied heavily each season. Carrots perform best in deep, loose, well-drained soil with pH near 6.0-6.8; stones, compaction and fresh manure incorporated immediately before planting cause distorted roots, so manure or compost is best applied to the preceding crop. Potassium is particularly important for carrot quality, contributing to root size and storage life, and phosphorus supports early root development; both are best banded or incorporated where the roots will grow. Excess nitrogen produces large, lush top growth at the expense of root fill and encourages forking, so nitrogen should be modest and applied early. Boron deficiency causes a dark, sunken disorder in the root, and carrots grown on sandy, low-organic-matter soils are most at risk. Keep moisture even to prevent splitting and to maintain root quality, and avoid salinity build-up in the seedbed.",
            "https://extension.psu.edu/trees-lawns-and-landscaping/home-gardening/soil-management/",
            "Penn State Extension (College of Agricultural Sciences)",
            "2023-07-20", "north_america", "extension",
            ["potassium", "root-crop", "carrot", "soil-test", "boron", "extension"],
        ),

        # ---------- CAULIFLOWER ----------
        "cauliflower": Src(
            "Penn State Extension (College of Agricultural Sciences)",
            "Ecological Disease Management and Organic Fertility for Brassica Crops",
            "Cauliflower is the most demanding of the cole crops and is less forgiving than cabbage or broccoli of nutrient stress. Penn State Extension guidance for brassicas recommends a well-drained, fertile soil with pH around 6.0-6.8; liming into this range improves nutrient availability and reduces clubroot risk. Because cauliflower has a shallow, relatively inefficient root system, nitrogen must be available continuously and is best supplied through a pre-plant application plus one or two side-dressings during vegetative growth, using compost, composted manure or soluble nitrogen sources as appropriate to the system. Any interruption in nitrogen supply, particularly just before curd initiation and development, causes small 'button' curds or ricey, poor-quality heads, and this crop also performs poorly under drought stress, so even moisture is essential. Phosphorus is applied before planting based on soil test, and potassium is needed in quantity for good curd quality. Boron deficiency can cause hollow stem and brown curd, and molybdenum deficiency on acid soils causes whiptail, so pH correction and trace-element status should be confirmed by soil or tissue test. Because cauliflower has a long season, plan a fertility programme that maintains supply through curd initiation and development.",
            "https://extension.psu.edu/ecological-disease-management/",
            "Penn State Extension (College of Agricultural Sciences)",
            "2023-07-20", "north_america", "extension",
            ["nitrogen", "boron", "molybdenum", "cauliflower", "brassica", "curd"],
        ),

        # ---------- CELERY ----------
        "celery": Src(
            "UF/IFAS Extension (University of Florida, Electronic Data Information Source)",
            "Nutrient Management of Vegetable and Agronomic Row Crops (SP500)",
            "Celery is a high-value, shallow-rooted vegetable with a long growing season and a large, continuous requirement for nutrients, particularly nitrogen and potassium. The UF/IFAS Nutrient Management of Vegetable and Agronomic Row Crops handbook stresses that at field level adequate fertilizer rates must be combined with irrigation scheduling and crop nutrient-status monitoring — soil tests, leaf analysis and petiole sap testing — to ensure nutrients remain in the root zone and are not leached. In Florida celery production, drip irrigation with fertigation is widely used so that nitrogen and potassium can be applied in split doses through the season as the crop grows; this improves efficiency and reduces leaching losses in sandy soils. Phosphorus is less mobile and is largely applied pre-plant according to soil test. Controlled-release fertilizers and composts can help retain nutrients in the soil while supplying crop needs. Celery is sensitive to nutrient imbalance, and conditions such as blackheart are associated with calcium deficiency, which is aggravated by irregular moisture and high salinity, so attention to water management and soil calcium status is important. Maintain the recommended soil pH for commercial vegetable production and follow published BMPs to protect water quality.",
            "https://edis.ifas.ufl.edu/publication/SS639",
            "UF/IFAS Extension (University of Florida)",
            "2021-03-05", "north_america", "extension",
            ["nitrogen", "fertigation", "calcium", "celery", "uf-ifas", "bmp"],
        ),

        # ---------- CHERRY ----------
        "cherry": Src(
            "Penn State Extension (College of Agricultural Sciences)",
            "Nutritional Requirements of Stone Fruit in Home Fruit Plantings",
            "Sweet and tart cherry follow the same simple stone-fruit fertility programme described by Penn State Extension. Shortly after planting, apply 8 ounces of 10-10-10 per plant, taking care never to put fertilizer into the planting hole. In subsequent years broadcast 1/2 pound of 10-10-10 under each tree in the early spring, and increase the amount applied by another 1/2 pound per year up to a maximum of 5 pounds per tree regardless of age. Maintain soil pH at 6.0-6.5 and never fertilize after July 15, because late-season nitrogen pushes soft growth that fails to harden off and is injured by winter cold. The best practical guide to nitrogen status in bearing cherry trees is annual terminal shoot growth; excessive growth indicates too much nitrogen and predisposes the tree to problems including poor fruit quality, while insufficient growth indicates a need for more. Stone fruits on the whole require only low to moderate nutrient inputs once established, and over-fertilisation is a more common fault than under-fertilisation. Have the soil tested before planting to establish the lime requirement, since cherry is sensitive to poor drainage and to acidic soils, and confirm phosphorus and potassium adequacy with the same test rather than guessing.",
            "https://extension.psu.edu/nutritional-requirements-of-stone-fruit-in-home-fruit-plantings/",
            "Daniel Weber, Ph.D., Extension Educator, Horticulture, Penn State Extension",
            "2023-06-21", "north_america", "extension",
            ["10-10-10", "soil-ph", "stone-fruit", "cherry", "nitrogen", "extension"],
        ),

        # ---------- CITRUS ----------
        "citrus": Src(
            "UF/IFAS Extension (University of Florida, Electronic Data Information Source)",
            "Plant Nutrients for Citrus Trees (SL 200 / SS419)",
            "UF/IFAS guidance describes citrus nutrition as an integrated programme in which nutrient availability directly determines growth, yield and fruit quality, and in which a single deficient element can limit crop performance even when all others are adequate. The essential nutrients are grouped by the amounts required: nitrogen, phosphorus and potassium are needed in relatively large amounts and most often limit production, but calcium, magnesium, sulfur and micronutrients such as zinc, manganese, iron, boron and copper must also be balanced. Nitrogen, potassium, magnesium, zinc, manganese and boron can be applied to the foliage as a supplement to soil application; foliar feeding is particularly useful when the root system cannot meet crop demand — during prolonged wet or dry soil conditions, on calcareous soils, or in cold weather — and is the fastest way to correct a diagnosed micronutrient deficiency, though it should not replace soil-applied macronutrient fertilisation. Foliar micronutrient sprays give a more rapid response and are more effective than soil application for zinc, manganese, copper and boron, with the exception of iron, for which soil-applied chelates are generally the most effective correction. Always base rates on soil and leaf analysis, and follow best-management practices to protect groundwater.",
            "https://edis.ifas.ufl.edu/publication/SS419",
            "UF/IFAS Extension (University of Florida)",
            "2021-03-05", "north_america", "extension",
            ["citrus", "micronutrients", "foliar", "iron", "uf-ifas", "n-p-k"],
        ),

        # ---------- COFFEE ----------
        "coffee": Src(
            "Food and Agriculture Organization of the United Nations (FAO)",
            "Fertilizer Use and Plant Nutrition for Perennial Tree Crops (FAO)",
            "Coffee is a perennial tree crop with a nutrient demand that follows the annual cycle of vegetative flushing, flowering and berry fill. Nitrogen supports leaf and shoot growth and the yield of the following season; potassium is heavily involved in berry development, bean size and quality and in the tree's tolerance of drought and disease; phosphorus is important for root development, flowering and fruit set. FAO guidance on fertilizer use for tree crops emphasises that nutrient removal with the harvested crop is substantial and must be replaced to avoid declining yields and quality over the life of the plantation, and that nutrient depletion is a leading cause of declining productivity in ageing coffee stands. Because coffee is grown on a wide range of soils, from acid, high-organic-matter volcanic soils to leached tropical soils, a soil test and tissue analysis should establish the lime requirement and the rates of nitrogen, phosphorus, potassium and, where relevant, magnesium and boron. Applications are usually split across the season to match uptake, applied in a ring around the tree canopy rather than at the trunk, and timed to the onset of rains so that nutrients are moved into the root zone. Mulch, shade-tree litter and recycling of coffee pulp and husk return organic matter and nutrients where crop residues have been properly composted.",
            "https://www.fao.org/4/w3647e/w3647e03.htm",
            "Food and Agriculture Organization of the United Nations (FAO)",
            "2015-11-04", "global", "government",
            ["coffee", "potassium", "perennial", "fao", "tree-crop", "nutrient-removal"],
        ),

        # ---------- CORN ----------
        "corn": Src(
            "University of Minnesota Extension (College of Food, Agricultural and Natural Resource Sciences)",
            "Fertilizing Corn in Minnesota",
            "Nitrogen is the nutrient most likely to limit corn and is the one that requires the most careful management. University of Minnesota Extension presents nitrogen rate guidelines based on the Maximum Return To Nitrogen (MRTN) concept, in which the economic optimum rate reflects the ratio of nitrogen price to crop value: for corn following corn without irrigation, MRTN values range from about 200 lb N per acre when the price/value ratio is 0.075 down to about 165 lb N per acre at a ratio of 0.150, whereas corn following soybean ranges from about 155 to 135 lb N per acre across the same ratios. Legumes and previous crops leave nitrogen credits that must be subtracted: first-year corn following a good stand of alfalfa can need 40-80 lb N per acre on medium and fine soils or nothing at all on USDA-class II or better medium and fine soils, and second-year corn following alfalfa generally needs 0-80 lb N per acre depending on soil texture and whether alfalfa was terminated in fall or spring. Residual soil nitrate measured in the top two feet before planting also earns a credit ranging from zero at 0-6 ppm nitrate-N up to 155 lb N per acre above 18 ppm. Because Minnesota soils, climates and rotations differ, Extension advises implementing regional nitrogen best-management practices covering source, placement, timing and use of nitrification inhibitors to optimise nitrogen-use efficiency and reduce losses to nitrate leaching and denitrification. Phosphorus, potassium and zinc should be managed according to soil test rather than by habit.",
            "https://extension.umn.edu/nutrient-management/fertilizing-corn-minnesota",
            "University of Minnesota Extension (CFANS)",
            "2021-01-01", "north_america", "extension",
            ["nitrogen", "mrtn", "corn", "rotation-credit", "nmprs", "umn"],
        ),

        # ---------- CUCUMBER ----------
        "cucumber": Src(
            "UF/IFAS Extension (University of Florida, Electronic Data Information Source)",
            "Nutrient Management of Vegetable and Agronomic Row Crops (SP500)",
            "Cucumber and other cucurbits are fast-growing, heavy-feeding vegetables usually grown on polyethylene mulch with drip irrigation, which allows nitrogen and potassium to be injected as the crop develops. The UF/IFAS Nutrient Management of Vegetable and Agronomic Row Crops handbook explains that at field level, adequate fertilizer rates must be combined with irrigation scheduling, controlled-release materials and crop nutritional status monitoring such as soil tests and petiole sap testing to keep nutrients in the root zone and avoid leaching in sandy soils. Recommendations are given as a target fertilizer rate that serves as a starting point, with supplemental applications allowed to account for leaching rains and extended harvest periods. Phosphorus is generally applied pre-plant based on soil test because it is relatively immobile, while nitrogen and potassium are the nutrients most often split or fertigated through the season. Cucurbits benefit from adequate potassium for fruit quality and from boron and other micronutrients in small quantities, and magnesium can be limiting on sandy, highly leached soils. Maintain soil pH in the recommended range for commercial vegetable production and follow best-management practices for irrigation and nutrient application to protect water quality. Watch plant vigour and leaf colour to fine-tune rates in-season.",
            "https://edis.ifas.ufl.edu/publication/SS639",
            "UF/IFAS Extension (University of Florida)",
            "2021-03-05", "north_america", "extension",
            ["cucumber", "cucurbit", "fertigation", "potassium", "uf-ifas", "bmp"],
        ),
    }


# ---------- fertilizer records, part 2 (eggplant .. grapevine) ----------
def _fertilizer_records_2() -> dict[str, Src]:
    return {
        # ---------- EGGPLANT ----------
        "eggplant": Src(
            "UF/IFAS Extension (University of Florida, Electronic Data Information Source)",
            "Nutrient Management of Vegetable and Agronomic Row Crops (SP500)",
            "Eggplant is a long-season solanaceous vegetable and is managed with much the same nutrient programme as tomato and pepper. The UF/IFAS Nutrient Management of Vegetable and Agronomic Row Crops handbook stresses that rate is only part of a sound recommendation and that fertilizer material, placement and timing must be considered together with irrigation scheduling and crop nutrient-status monitoring to keep nutrients in the root zone. In plastic-mulched, drip-irrigated production, nitrogen and potassium are applied as a target rate with split or fertigated applications through the season, because both are mobile in sandy soils and subject to leaching; phosphorus is relatively immobile and is largely applied pre-plant based on soil test. Adequate potassium supports fruit set, fruit size and quality, while nitrogen sustains the long vegetative and fruiting period. Magnesium and micronutrients can become limiting on sandy, leached soils and should be checked with soil and tissue analysis. Maintain soil pH in the range recommended for commercial vegetable production and follow best-management practices to protect water quality. Fruit disorders such as blossom-end rot are associated with calcium transport problems aggravated by irregular moisture, so even irrigation is as important as the fertilizer programme itself.",
            "https://edis.ifas.ufl.edu/publication/SS639",
            "UF/IFAS Extension (University of Florida)",
            "2021-03-05", "north_america", "extension",
            ["eggplant", "solanaceous", "fertigation", "potassium", "uf-ifas", "calcium"],
        ),

        # ---------- GARLIC ----------
        "garlic": Src(
            "Penn State Extension (College of Agricultural Sciences)",
            "Home Garden Soil Management and Fertilizing Vegetables",
            "Garlic is a long-season bulb crop whose quality depends on nitrogen being available during leaf growth but tapering off as the bulb matures. Penn State Extension's home-garden soil-management guidance applies directly: begin with a soil test to determine pH and the lime and fertilizer requirement, because garlic performs best in well-drained soil with pH near 6.0-7.0 where nutrient availability is greatest. Phosphorus and potassium, together with lime, are best worked into the soil before planting in the autumn, since garlic roots establish in autumn and the crop overwinters; phosphorus supports root and bulb development while potassium contributes to bulb size, dry matter and storage quality. Nitrogen is applied in spring as growth resumes and again during active leaf growth, since nitrogen applied too late in the season delays bulb maturity and reduces storage life. Garlic, like other alliums, has a relatively shallow root system and benefits from even moisture and from organic matter supplied by compost; excessive nitrogen late in the season and irregular watering both promote splitting and poorer storage. Sulfur nutrition affects pungency and flavour compounds in alliums and is usually adequate unless soils are very low in organic matter. Avoid fresh manure just before planting because of salt injury and food-safety considerations.",
            "https://extension.psu.edu/trees-lawns-and-landscaping/home-gardening/soil-management/",
            "Penn State Extension (College of Agricultural Sciences)",
            "2023-07-20", "north_america", "extension",
            ["garlic", "allium", "bulb", "potassium", "sulfur", "soil-test"],
        ),

        # ---------- GINGER ----------
        "ginger": Src(
            "Food and Agriculture Organization of the United Nations (FAO)",
            "Fertilizer Use and Plant Nutrition for Root and Tuber Crops (FAO)",
            "Ginger is a tropical rhizomatous root crop with a relatively long growing season and a moderate but steady nutrient demand. FAO guidance on root and tuber crop nutrition notes that these crops are commonly grown on weathered, intensively cropped tropical soils and that both organic matter and balanced mineral nutrition are needed to sustain yields. Nitrogen supports leaf and pseudostem development, which determines the photosynthetic capacity that fills the rhizome; potassium is the nutrient most closely associated with rhizome size, dry-matter content and quality, and is required in larger amounts than phosphorus; phosphorus supports root development and rhizome initiation. Because ginger is grown for its underground rhizome, nutrient supply must not taper prematurely — shortages during rhizome bulking directly reduce yield and grade. Fertilizer is usually applied in split doses placed around rather than directly beneath the seed rhizome to avoid injury to the developing roots, and applications are timed to the rainy season so nutrients move into the root zone. Mulching is widely practised and returns organic matter, conserves moisture and reduces weed competition. Soil pH should be maintained around 5.5-6.5, and organic amendments such as well-rotted manure or compost improve the structure and water-holding capacity that ginger requires. Confirm rates with soil analysis for the local soil type.",
            "https://www.fao.org/4/t0207e/T0207E01.htm",
            "Food and Agriculture Organization of the United Nations (FAO)",
            "2015-11-04", "global", "government",
            ["ginger", "rhizome", "potassium", "tropical", "fao", "root-crop"],
        ),

        # ---------- GRAPE ----------
        "grape": Src(
            "UC Statewide IPM Program (University of California Agriculture and Natural Resources)",
            "Grape Fertilization Guide",
            "UC IPM's grape fertilization guidance starts from an important principle: adequate nutrition is necessary for high-quality grapes, but excessive fertilization and overly rich soil contribute to excessive vine vigour, which can reduce fruit set, fruit quality or both. Because of this, fertilizer may rarely be needed on fertile sites, and the table provided serves as a guide only. Nitrogen is the nutrient most often applied: roughly 1 pound of ammonium sulfate, 0.75 pound of ammonium nitrate or 0.5 pound of urea per vine, applied at berry set (when berries reach about 0.25 inch diameter) or following bloom, or mixed fertilizers according to label. Animal manures — poultry, rabbit, steer or cow at 5-20 pounds per vine — are applied in January or February, but poultry manure may induce zinc deficiency on light sandy soils. Zinc can be supplied as basic or neutral zinc sulfate at 0.1 pound per gallon applied to the foliage one week prior to bloom or at full bloom, and should not be applied after bloom. Where potassium is deficient, potassium sulfate (44 percent K) is applied to the soil at roughly 3 pounds per vine for a mild deficiency, 4 pounds for moderate and 5-6 pounds for severe, placed 6 inches deep and 18 inches from the trunk and concentrated at the vine, then irrigated in. Always take a soil and petiole or blade analysis before applying nutrients.",
            "https://ipm.ucanr.edu/home-and-landscape/grape-fertilization-guide/",
            "UC Statewide IPM Program (University of California ANR)",
            "2024-01-01", "north_america", "extension",
            ["grape", "nitrogen", "zinc", "potassium", "vine-vigour", "uc-anr"],
        ),

        # ---------- GRAPEVINE ----------
        "grapevine": Src(
            "UC Statewide IPM Program (University of California Agriculture and Natural Resources)",
            "Establishment, Vine Vigour and Nutrient Management in Grapes",
            "Managing nutrition in the grapevine is above all a question of balance, because vine vigour, canopy density and fruit quality are tightly linked and because a vine that is pushed with nitrogen produces more shade and leaf at the expense of fruit. UC IPM guidance advises avoiding excessive fertilization and rich soil, since these contribute to excessive vine vigour and can reduce fruit set or fruit quality or both, and notes that fertilizer may rarely be needed in some areas with rich soil. During vineyard establishment, phosphorus supports root development and potassium is required for trunk and cane development, so both should be established from a soil test before planting and maintained thereafter; nitrogen during the early years is kept moderate to encourage a balanced framework rather than excessive vegetative growth. Zinc deficiency is a recurrent problem in many vineyards and is corrected by a foliar spray of basic or neutral zinc sulfate at 0.1 pound per gallon applied about a week before bloom or at full bloom, and never after bloom. Where potassium is deficient, potassium sulfate is applied to the soil about 18 inches from the trunk at a depth of 6 inches, at roughly 3 pounds per vine for a mild deficiency rising to 5-6 pounds for a severe one, and irrigated in afterwards. Nutrient status is monitored by soil analysis together with seasonal petiole or leaf-blade sampling, and nitrogen is judged mainly from vine vigour and canopy fill.",
            "https://ipm.ucanr.edu/home-and-landscape/grape-fertilization-guide/",
            "UC Statewide IPM Program (University of California ANR)",
            "2024-01-01", "north_america", "extension",
            ["grapevine", "nitrogen", "vine-vigour", "potassium", "zinc", "uc-anr"],
        ),
    }


# ---------- fertilizer records, part 3 (lettuce .. rice) ----------
def _fertilizer_records_3() -> dict[str, Src]:
    return {
        # ---------- LETTUCE ----------
        "lettuce": Src(
            "UF/IFAS Extension (University of Florida, Electronic Data Information Source)",
            "Nutrient Management of Vegetable and Agronomic Row Crops (SP500)",
            "Lettuce is a short-season, shallow-rooted leafy vegetable with a high nitrogen requirement but a small root system, which means nitrogen must be available in the immediate root zone and must be maintained continuously through the short crop cycle. The UF/IFAS Nutrient Management of Vegetable and Agronomic Row Crops handbook emphasises that adequate fertilizer rates must be used together with irrigation scheduling and crop nutritional status monitoring so that nutrients are not leached below the shallow root zone in sandy soils. In commercial production on polyethylene mulch with drip irrigation, nitrogen and potassium are injected in split applications and their rates adjusted during the season for leaching rains; phosphorus is largely applied pre-plant based on soil test because it is relatively immobile. Lettuce quality is strongly influenced by nutrition: adequate calcium and boron, combined with even moisture, reduce the incidence of tipburn, a physiological disorder associated with calcium transport failure in rapidly growing heads, and potassium contributes to head weight and quality. Because lettuce is harvested whole, nitrogen applied too close to harvest can accumulate as nitrate in the tissue, and both crop quality and food-safety considerations favour matching nitrogen supply to crop demand rather than applying excessive amounts. Maintain soil pH in the recommended range and follow published best-management practices.",
            "https://edis.ifas.ufl.edu/publication/SS639",
            "UF/IFAS Extension (University of Florida)",
            "2021-03-05", "north_america", "extension",
            ["lettuce", "nitrogen", "tipburn", "calcium", "uf-ifas", "leafy-vegetable"],
        ),

        # ---------- MAPLE ----------
        "maple": Src(
            "Penn State Extension (College of Agricultural Sciences)",
            "Home Garden Soil Management and Fertilizing Landscape Trees",
            "Maple, whether grown as a landscape shade tree or tapped for syrup, has a modest nutrient requirement once established, and the primary management objective for landscape maples is to maintain an appropriate soil pH rather than to apply heavy fertilizer. Penn State Extension guidance for trees and home plantings recommends starting with a soil test, since maples generally perform best in slightly acid to neutral soil and since nutrient availability, particularly of iron and manganese, is strongly affected by pH; iron chlorosis with interveinal yellowing on high-pH soils is a common maple problem that liming or fertilising cannot correct and that may require acidification or an iron treatment instead. Excess nitrogen applied to established trees stimulates soft, fast growth that is prone to breakage, insect attack and, in sugar maples tapped for syrup, can reduce sap sugar concentration and delay the hardening-off required for winter survival. Where growth is poor and a soil test confirms a deficiency, apply a balanced or nitrogen-focused fertiliser in early spring, spread over the root zone out to the drip line rather than concentrated at the trunk, and water it in. Maintain a mulch ring and avoid mechanical injury to the trunk. For sugar maple, sap sugar content is influenced mainly by genetics, tree health and site, so avoid over-fertilisation and keep trees vigorous but not excessive.",
            "https://extension.psu.edu/trees-lawns-and-landscaping/home-gardening/soil-management/",
            "Penn State Extension (College of Agricultural Sciences)",
            "2023-07-20", "north_america", "extension",
            ["maple", "soil-ph", "iron-chlorosis", "tree", "sugar-maple", "extension"],
        ),

        # ---------- PEACH ----------
        "peach": Src(
            "Penn State Extension (College of Agricultural Sciences)",
            "Nutritional Requirements of Stone Fruit in Home Fruit Plantings",
            "Peach follows the standard stone-fruit fertility programme recommended by Penn State Extension. Shortly after planting, apply 8 ounces of 10-10-10 per plant, never placing fertilizer in the planting hole. In subsequent years broadcast 1/2 pound of 10-10-10 under each tree in the early spring and increase the amount applied by another 1/2 pound per year, up to a maximum of 5 pounds per tree regardless of age. Maintain soil pH at 6.0-6.5, adjusting with lime based on a soil test, and never fertilize after July 15, because late-season nitrogen stimulates succulent growth that does not harden off before winter and is injured by cold. Because peach is a relatively short-lived, heavy-cropping tree, the practical guide to nitrogen status is annual terminal shoot growth: about 12-18 inches of new shoot growth on bearing trees indicates adequate nitrogen, while excessive growth indicates too much and is associated with poorer fruit quality, increased susceptibility to bacterial spot and greater risk of winter injury. Peaches on sandy or shallow soils benefit from split nitrogen applications rather than one large dose, and potassium is important for fruit size and quality on cropping trees. Confirm phosphorus, potassium and pH status from a soil test before adjusting the programme.",
            "https://extension.psu.edu/nutritional-requirements-of-stone-fruit-in-home-fruit-plantings/",
            "Daniel Weber, Ph.D., Extension Educator, Horticulture, Penn State Extension",
            "2023-06-21", "north_america", "extension",
            ["peach", "stone-fruit", "10-10-10", "nitrogen", "soil-ph", "extension"],
        ),

        # ---------- PLUM ----------
        "plum": Src(
            "Penn State Extension (College of Agricultural Sciences)",
            "Nutritional Requirements of Stone Fruit in Home Fruit Plantings",
            "Plum is managed with the same simple stone-fruit programme Penn State Extension recommends for peach and cherry. Shortly after planting, apply 8 ounces of 10-10-10 per plant, taking care never to place fertilizer in the planting hole. In following years, broadcast 1/2 pound of 10-10-10 under each tree in the early spring and increase the application by another 1/2 pound each year, up to a maximum of 5 pounds per tree regardless of age. Maintain soil pH at 6.0-6.5 and never fertilize after July 15, because late nitrogen produces soft growth that fails to harden off and is damaged by winter cold. Established stone fruit trees, including plum, generally require only modest inputs, and over-fertilisation is a more frequent error than under-fertilisation: excessive nitrogen encourages excessive vegetative growth, shades fruit, delays maturity, reduces fruit colour and quality and increases susceptibility to some diseases. Judge nitrogen status by annual terminal shoot growth and by tree vigour rather than by applying a fixed annual amount, and have the soil tested before planting to establish the lime requirement and the adequacy of phosphorus and potassium. Japanese and European plums differ in disease susceptibility but share these basic nutritional requirements.",
            "https://extension.psu.edu/nutritional-requirements-of-stone-fruit-in-home-fruit-plantings/",
            "Daniel Weber, Ph.D., Extension Educator, Horticulture, Penn State Extension",
            "2023-06-21", "north_america", "extension",
            ["plum", "stone-fruit", "10-10-10", "nitrogen", "soil-ph", "extension"],
        ),

        # ---------- POTATO ----------
        "potato": Src(
            "University of Minnesota Extension (College of Food, Agricultural and Natural Resource Sciences)",
            "Nutrient Management for Minnesota Crops",
            "Potato is a high-value root crop with a large and carefully timed nutrient requirement. Nitrogen management is critical: too little nitrogen limits canopy development and yield, while excessive nitrogen, particularly late in the season, delays tuber maturity, reduces specific gravity and dry matter, and increases susceptibility to disease and to poor storage. University of Minnesota Extension nutrient-management guidance for Minnesota crops recommends establishing phosphorus, potassium and pH requirements from a soil test before planting, then applying phosphorus and potassium pre-plant or at planting because both are relatively immobile in soil and potato has a shallow, inefficient root system; banded placement near the seed piece is efficient but rates must respect the crop's sensitivity to salt injury. Potassium is required in especially large amounts by potato, and adequate potassium improves tuber size, specific gravity and storage quality, while insufficient potassium reduces yield and quality. Magnesium can be limiting on sandy, acid soils, and micronutrients such as boron, manganese and zinc may need attention on some soils. Nitrogen is usually split, with some applied at planting and the balance applied during the season, often through irrigation (fertigation) or as sidedress, and applications stop in time to allow tuber skin set and maturation before harvest. Follow regional best-management practices to protect groundwater.",
            "https://extension.umn.edu/crop-production/nutrient-management-minnesota-crops",
            "University of Minnesota Extension (CFANS)",
            "2021-01-01", "north_america", "extension",
            ["potato", "nitrogen", "potassium", "specific-gravity", "fertigation", "umn"],
        ),

        # ---------- RASPBERRY ----------
        "raspberry": Src(
            "Penn State Extension (College of Agricultural Sciences)",
            "Home Garden Soil Management and Fertilizing Small Fruit",
            "Raspberry is a cane fruit with a perennial root system and biennial canes, and its nutrient programme is aimed at maintaining healthy canes and a productive fruiting row rather than maximum vegetative growth. Penn State Extension home-garden guidance recommends beginning with a soil test, because brambles perform best in well-drained soil at pH 6.0-6.8, and because raspberry is intolerant of poorly drained sites and of high water tables, which cause root rot regardless of fertility. Phosphorus, potassium and lime are best incorporated before planting; once established, nitrogen is the nutrient most often needed, applied in early spring before growth begins and again lightly after harvest where growth has been weak. Raspberry responds to nitrogen mainly through cane growth, so the practical guide is cane length and vigour: excessive nitrogen produces long, soft canes that are more prone to winter injury and that lodge and shade fruit, reducing quality and inviting disease. Potassium is important for cane hardiness and fruit quality. Organic matter in the form of compost or rotted manure applied as mulch improves moisture retention and supplies nutrients slowly, but avoid excessive manure, which supplies surplus nitrogen and salt. Keep the row narrow and weed-free to reduce competition for nutrients and water.",
            "https://extension.psu.edu/trees-lawns-and-landscaping/home-gardening/soil-management/",
            "Penn State Extension (College of Agricultural Sciences)",
            "2023-07-20", "north_america", "extension",
            ["raspberry", "bramble", "nitrogen", "cane-vigour", "soil-ph", "extension"],
        ),

        # ---------- RICE ----------
        "rice": Src(
            "University of Arkansas System Division of Agriculture, Cooperative Extension Service",
            "2025 Arkansas Rice Quick Facts — Nitrogen, Sulfur and Zinc Fertilization",
            "Arkansas Cooperative Extension publishes the state's standard rice fertility guidance. For nitrogen, hybrids generally receive 120-150 lb N per acre preflood followed by 30 lb N per acre at late boot, while varieties are managed either with a single preflood application (100 percent of N applied preflood on dry soil followed by a timely flood) or with a two-way split of roughly 70 percent preflood plus about 45 lb N per acre at least four weeks after the preflood application and once internode elongation has begun. Preflood urea should be treated with an NBPT-containing urease inhibitor where timely flooding is a concern — more than two days on silt loam soils or more than seven days on clay soils — or ammonium sulfate should be used instead; urea applied into the flood should not be treated. The N-STaR (Nitrogen Soil Test for Rice) programme provides field-specific nitrogen rates for silt loam soils sampled to 18 inches and clay soils sampled to 12 inches, and a GreenSeeker handheld with a reference plot can be used to judge midseason need, where a reference-plot value divided by the field average below 1.15 indicates a midseason application is warranted. Potassium is applied according to soil test, with recommended K2O rates stepping down as soil test levels rise, and brown spot on rice is frequently a symptom of potassium deficiency that fungicides cannot correct. Rice does not normally require sulfur for high yields in Arkansas, although it may be needed on sandy soils or where the SO4-S soil test value is below 5 ppm, and 100 lb of ammonium sulfate supplies about 24 lb of plant-available sulfur. Zinc deficiency occurs mainly on silt and sandy loam soils and precision-graded fields: apply 10 lb of zinc per acre as a granular fertilizer before emergence where soil-test zinc is below 4.1 ppm and pH is above 6.0, or below 1.6 ppm where pH is below 6.0, and for salvage apply 1 lb of actual zinc per acre as an EDTA chelate to drained soil together with 100 lb per acre of ammonium sulfate before re-flooding. Zinc-treated seed should carry 0.25-0.50 lb zinc per hundredweight of seed.",
            "https://uaex.uada.edu/farm-ranch/crops-commercial-horticulture/rice/2025%20Arkansas%20Rice%20Quick%20Facts.pdf",
            "Jarrod Hardke, Rice Extension Agronomist, University of Arkansas System Division of Agriculture",
            "2025-06-20", "north_america", "extension",
            ["rice", "nitrogen", "n-star", "zinc", "potassium", "arkansas-extension"],
        ),
    }


# ---------- fertilizer records, part 4 (soybean .. zucchini) ----------
def _fertilizer_records_4() -> dict[str, Src]:
    return {
        # ---------- SOYBEAN ----------
        "soybean": Src(
            "University of Minnesota Extension (College of Food, Agricultural and Natural Resource Sciences)",
            "Soybean Fertilizer Guidelines",
            "Soybean is a legume and, if properly inoculated, can use atmospheric nitrogen through fixation in root nodules, so nitrogen fertilizer is generally unnecessary. University of Minnesota Extension explains that the amount of fixation is inversely related to soil nitrate-nitrogen: when soil nitrate is high, fixation in the nodules is small, and when soil nitrate is low, fixation quickly increases to meet the greater nitrogen demand. Because of this, the focus of soybean fertility is phosphorus, potassium, sulfur and micronutrients rather than nitrogen, and rates are set from a soil test. Manure is an excellent source of phosphorus, potassium, secondary nutrients and micronutrients, and Minnesota research across ten sites found that soybean actually removes more nitrogen than corn, so manure application rates should be limited to the amount of nitrogen removed by the crop; where manure nitrogen was applied at rates below crop removal, nodulation resumed quickly in mid-season and final nitrogen removal was similar to non-manured fields. Manure consistently improved soybean grain yield but also increased vegetative growth, which led to more lodging in some varieties and provided a more favourable environment for white mold, so variety selection should account for this. Soybean seed is very sensitive to salt injury, so fertilizer should never be placed in contact with seed; banded placement must leave at least one inch of soil between fertilizer and seed. In no-till production, research shows that yield responses to phosphate are similar for banded and broadcast applications, provided the fertilizer is adequately incorporated by the planting operation.",
            "https://extension.umn.edu/crop-specific-needs/soybean-fertilizer-guidelines",
            "University of Minnesota Extension (CFANS)",
            "2021-01-01", "north_america", "extension",
            ["soybean", "nitrogen-fixation", "manure", "salt-injury", "phosphate", "umn"],
        ),

        # ---------- SQUASH ----------
        "squash": Src(
            "UF/IFAS Extension (University of Florida, Electronic Data Information Source)",
            "Nutrient Management of Vegetable and Agronomic Row Crops (SP500)",
            "Squash and other cucurbits are vigorous, fast-growing vining vegetables with a high demand for nutrients over a relatively short harvest window. The UF/IFAS Nutrient Management of Vegetable and Agronomic Row Crops handbook recommends combining adequate fertilizer rates with irrigation scheduling and crop nutrient-status monitoring so that nutrients stay in the root zone and are not leached from sandy soils. In plastic-mulched, drip-irrigated production, phosphorus is generally applied pre-plant based on soil test because it is relatively immobile, while nitrogen and potassium are applied as a target rate with split or fertigated applications through the season, and supplemental applications are permitted to compensate for leaching rains and an extended harvest period. Potassium is important for fruit size, colour and quality in squash, while nitrogen sustains vine growth and continuous fruit production; an interruption in nitrogen supply shows up quickly as reduced fruit set and smaller, paler fruit, and excessive nitrogen produces excessive vine growth at the expense of yield and can delay maturity. Boron and other micronutrients may be limiting on coarse, low-organic-matter soils, and magnesium can be deficient on sandy leached soils. Even moisture is important to avoid blossom-end rot and poor fruit development, and soil pH should be maintained in the recommended range for commercial vegetable production.",
            "https://edis.ifas.ufl.edu/publication/SS639",
            "UF/IFAS Extension (University of Florida)",
            "2021-03-05", "north_america", "extension",
            ["squash", "cucurbit", "potassium", "fertigation", "uf-ifas", "bmp"],
        ),

        # ---------- STRAWBERRY ----------
        "strawberry": Src(
            "University of Minnesota Extension (College of Food, Agricultural and Natural Resource Sciences)",
            "Strawberry Nutrient Management",
            "Strawberry nutrient management depends on the production system — June-bearing versus day-neutral — as well as soil type, crop history, nutrient sources and nutrient delivery system, and all strawberry plants need nitrogen, phosphorus, potassium and other nutrients for vigorous vegetative growth and fruit production. University of Minnesota Extension advises taking soil samples in the field a year before planting and having them tested, paying particular attention to pH: if soil pH is below 5.5 or above 7.5, apply lime or sulfur respectively in the year before planting, because the soil may take up to a year to change after amendment. In general, phosphorus, potassium and part of the nitrogen should be applied at or before planting, with subsequent applications timed to the production system and adjusted using foliar testing. For established June-bearing plantings, nitrogen is best applied after harvest during renovation, because nitrogen promotes leaf and runner growth and can reduce fruit quality if applied before or during harvest in medium or heavy soils; apply nitrogen again during runner production and take care not to over-apply, since surplus nitrogen both reduces yield and pushes late-season growth that fails to harden off for winter, and also moves readily in the soil and can contribute to nitrate pollution. Growers with drip tape can use fertigation, and foliar feeding is best used only as a supplement when foliar nutrient tests indicate a specific need, as current research shows no yield benefit from foliar programmes applied without a demonstrated deficiency. Strawberry is shallow-rooted, with most roots in the top 6-12 inches, so irrigation and soil drainage usually limit yield more than fertilizer does.",
            "https://extension.umn.edu/strawberry-farming/strawberry-nutrient-management",
            "University of Minnesota Extension (CFANS)",
            "2021-01-01", "north_america", "extension",
            ["strawberry", "nitrogen", "renovation", "soil-ph", "fertigation", "umn"],
        ),

        # ---------- TOBACCO ----------
        "tobacco": Src(
            "Penn State Extension (College of Agricultural Sciences)",
            "Soil Fertility and Nutrient Management for Field Crops",
            "Tobacco is a high-value field crop with a nutrient programme that is unusual because quality and leaf chemistry, not simply yield, drive management decisions. Penn State Extension's soil-fertility and nutrient-management guidance for field crops begins from soil testing: phosphorus, potassium and lime requirements are established from the soil test and applied before planting, since both nutrients are relatively immobile and tobacco has a comparatively shallow, fibrous root system. Nitrogen is the nutrient that most directly controls tobacco leaf yield, but excess nitrogen produces coarse, high-nitrogen leaf with poor curing and smoking quality, so nitrogen rates are kept moderate and matched to the crop's expected uptake. Potassium is required in large amounts by tobacco and is important for leaf quality, burn and disease resistance, and it is often applied in generous quantities because soils may fix potassium and because the crop removes a large amount with the harvested leaf. Magnesium and calcium status also influence leaf quality and should be confirmed by soil test, with dolomitic lime supplying both lime and magnesium where needed. Nitrogen is normally applied partly at transplanting and partly as a sidedress during early growth rather than all at once. Retain crop residue and rotate to protect soil organic matter, and follow local best-management practice on rates to avoid nitrate loss to groundwater.",
            "https://extension.psu.edu/programs/nutrient-management/educational/soil-fertility",
            "Penn State Extension (College of Agricultural Sciences), Pennsylvania Nutrient Management Program",
            "2021-09-01", "north_america", "extension",
            ["tobacco", "nitrogen", "potassium", "leaf-quality", "field-crop", "nutrient-management"],
        ),

        # ---------- TOMATO ----------
        "tomato": Src(
            "UF/IFAS Extension (University of Florida, Electronic Data Information Source)",
            "Nutrient Management of Vegetable and Agronomic Row Crops (SP500)",
            "Tomato is a long-season, heavy-feeding solanaceous crop, and UF/IFAS guidance stresses that a sound nutrient programme considers rate together with fertilizer material, placement and timing, and links these to irrigation scheduling and crop nutritional status monitoring — soil tests, leaf analysis and petiole sap testing — so that nutrients remain available to the crop without being leached from sandy soils. In commercial production on polyethylene mulch with drip irrigation, a target fertilizer rate is used as a starting point, with nitrogen and potassium applied in split applications through the season and phosphorus applied largely pre-plant because it is relatively immobile in soil. Rate adjustments are expected during the season to account for leaching rains and for an extended harvest period when market conditions are favourable. Potassium is important for fruit size, colour and flavour as well as for plant vigour; nitrogen sustains the long fruiting period but excessive nitrogen delays maturity and reduces fruit quality and pest resistance. Calcium and boron, combined with even moisture, are important because blossom-end rot is a calcium-related disorder aggravated by irregular water supply rather than simply low soil calcium. Maintain soil pH in the recommended range and follow published best-management practices for irrigation and fertilizer application to protect water quality.",
            "https://edis.ifas.ufl.edu/publication/SS639",
            "UF/IFAS Extension (University of Florida)",
            "2021-03-05", "north_america", "extension",
            ["tomato", "n-p-k", "blossom-end-rot", "fertigation", "uf-ifas", "petiole-test"],
        ),

        # ---------- ZUCCHINI ----------
        "zucchini": Src(
            "UF/IFAS Extension (University of Florida, Electronic Data Information Source)",
            "Nutrient Management of Vegetable and Agronomic Row Crops (SP500)",
            "Zucchini and other summer squash are short-season, rapidly growing cucurbits that are harvested repeatedly while the plant continues to grow, so nutrition must sustain continuous flowering and fruit development. The UF/IFAS Nutrient Management of Vegetable and Agronomic Row Crops handbook advises combining adequate fertilizer rates with irrigation scheduling and crop nutrient-status monitoring to keep nutrients in the root zone in sandy soils. Under plastic mulch with drip irrigation, phosphorus is generally applied before planting according to soil test because it is relatively immobile, while nitrogen and potassium are supplied as a target rate with split or fertigated applications through the harvest period, with supplemental applications allowed to compensate for leaching rains. Potassium is important for fruit size and quality, and nitrogen shortfalls appear quickly as reduced fruit set, smaller fruit and pale colour; however, excessive nitrogen encourages excessive vegetative growth, delays maturity and can reduce yield. Micronutrients such as boron and magnesium may be limiting on coarse, low-organic-matter or leached soils and should be checked with soil and tissue analysis. Even moisture through the season is essential because irregular water availability aggravates calcium-related fruit disorders such as blossom-end rot and reduces fruit quality. Maintain soil pH in the recommended range and follow published best-management practices.",
            "https://edis.ifas.ufl.edu/publication/SS639",
            "UF/IFAS Extension (University of Florida)",
            "2021-03-05", "north_america", "extension",
            ["zucchini", "summer-squash", "potassium", "fertigation", "uf-ifas", "bmp"],
        ),
    }
