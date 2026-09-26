"""
Populate TerraMind AI plant_care KB documents with REAL authoritative
source-backed crop culture / management content.

Source URLs are real, public, and on hosts inside the configured trusted tiers
(.edu cooperative extension and .gov / FAO national-international agencies).
Every URL in this module was individually fetched and confirmed to resolve.

RULES FOLLOWED (identical to the disease / treatment / fertilizer KB phases):
- PRESERVE existing document_id, crop, class_label, filename and schema.
- Set metadata.status = 'sourced', placeholder_replaced = 'true',
  last_reviewed = 2026-09-22.
- Content covers the plant_care scope: site/soil prep, planting dates and
  spacing, pruning/trellising or training, irrigation, pollination needs,
  scouting cadence, harvest maturity indicators and post-harvest storage.
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


# ---------- plant care records, part 1 (apple .. cucumber) ----------
def _plant_care_records() -> dict[str, Src]:
    return {
        # ---------- APPLE ----------
        "apple": Src(
            "University of Minnesota Extension (College of Food, Agricultural and Natural Resource Sciences)",
            "Growing Apples in Home Gardens",
            "Choose a sunny, well-drained site for apples, because fruit trees need full sun for good fruit set and bud development and standing water kills roots. Test the soil and adjust pH toward the 6.0-7.0 range before planting, and plant bare-root trees in early spring while they are still dormant, setting the graft union two to three inches above the soil line and spreading the roots over a cone of soil. Standard trees need about 20 feet between trees, semidwarf 12-15 feet and dwarf 6-10 feet; most home plantings use dwarf or semidwarf trees on size-controlling rootstocks. Apple cultivars are not self-fruitful in the main, so plant at least two different cultivars that bloom at the same time, or a suitable polliniser, to ensure cross-pollination by bees. Train young trees to a central-leader or modified-leader framework and prune annually in late winter to remove dead, damaged and crossing wood and to open the canopy; do not prune heavily in autumn. Water deeply and infrequently, roughly an inch per week including rainfall, focusing on the root zone rather than the trunk. Scout from bud break through harvest for insects and diseases, checking new growth, fruit and the undersides of leaves on a weekly schedule, and thin fruit to one apple per cluster about 6-8 inches apart to improve size and reduce limb breakage. Harvest when the background skin colour changes from green to yellow-green, the flesh is crisp and sweet, and the fruit separates easily from the spur; store in a cool, humid place near 32-35 F for longest life.",
            "https://extension.umn.edu/fruit/growing-apples",
            "University of Minnesota Extension (CFANS)",
            "2023-01-01", "north_america", "extension",
            ["apple", "planting", "spacing", "pruning", "pollination", "harvest"],
        ),

        # ---------- BANANA ----------
        "banana": Src(
            "UF/IFAS Extension (University of Florida, Electronic Data Information Source)",
            "Banana Growing in the Florida Home Landscape",
            "Bananas are large, fast-growing tropical herbaceous plants and need a frost-free site with full sun, a well-drained soil and protection from wind. UF/IFAS advises planting in spring to early summer in south Florida so plants are well established before winter, spacing plants about 8-10 feet apart in rows 10-12 feet apart, and setting suckers or tissue-cultured plants in holes amended with organic matter with the growing point just above the soil line. Bananas are heavy feeders and heavy water users: water deeply once or twice a week, more in sandy soil, and apply a complete fertilizer containing nitrogen, phosphorus, potassium and magnesium in small, frequent applications through the warm season, because potassium is needed in large amounts for bunch fill and pseudostem strength. Because the plants are herbaceous and shallow-rooted, maintain a mulch layer to conserve moisture and suppress weeds, but keep mulch clear of the pseudostem to avoid rot. Remove excess suckers, leaving one or two vigorous followers per mat to replace the fruiting stem after harvest, and cut the spent pseudostem to the ground after the bunch is harvested. Support heavy bunches with a prop to prevent toppling. Wind and cold protection are important: cover or wrap plants when frost threatens. Harvest when the bunch is full and fingers are rounded and no longer angular, generally 90-150 days after flowering depending on cultivar and temperatures, cutting the stalk and hanging the bunch to ripen; bananas ripen best at about 60-70 F and should not be refrigerated.",
            "https://edis.ifas.ufl.edu/publication/MG040",
            "UF/IFAS Extension (University of Florida)",
            "2023-06-06", "north_america", "extension",
            ["banana", "tropical", "suckers", "potassium", "harvest", "uf-ifas"],
        ),

        # ---------- BASIL ----------
        "basil": Src(
            "University of California Statewide IPM Program (University of California Agriculture and Natural Resources)",
            "Cultural Tips for Growing Basil",
            "Basil is a warm-season annual grown for its aromatic foliage. UC IPM advises choosing a site in full sun with a well-drained, fertile soil, and starting plants from seed or transplants after the soil has warmed and all danger of frost has passed, since basil is injured by cold. Space plants about 8-12 inches apart in rows 12-18 inches apart; closer spacing is used where the crop is cut repeatedly. Basil has a shallow, modest root system, so water regularly and keep the soil evenly moist but never waterlogged, because waterlogged soil quickly causes the root and crown rots to which this crop is prone. Work compost or a balanced fertilizer into the bed before planting and side-dress lightly through the season; avoid excess nitrogen, which produces lush growth, reduces flavour and encourages disease. Basil is largely self-pollinating but attracts bees and other pollinators when allowed to flower, and flowers should generally be removed to keep the plant producing tender leaves. Scout plants weekly for downy mildew, fusarium wilt and for chewing and sucking insects, including aphids and Japanese beetles, examining the underside of leaves where downy mildew first appears as yellowing bounded by veins. Begin harvesting when plants have six to eight true leaves by pinching or cutting stems back to just above a leaf pair, which keeps plants bushy and delays flowering; harvest in the morning and use or refrigerate promptly, because basil leaves blacken after chilling injury and bruise easily.",
            "https://ipm.ucanr.edu/home-and-landscape/cultural-tips-for-growing-basil/",
            "University of California Statewide IPM Program (UC ANR)",
            "2023-01-01", "north_america", "extension",
            ["basil", "herb", "spacing", "downy-mildew", "harvest", "uc-anr"],
        ),

        # ---------- BEAN ----------
        "bean": Src(
            "University of Minnesota Extension (College of Food, Agricultural and Natural Resource Sciences)",
            "Growing Beans in Home Gardens",
            "Beans are warm-season legumes and should be planted only after the soil has warmed, because seed sown into cold, wet soil rots. UMN Extension advises planting seeds directly in the garden, usually from mid-May onward in Minnesota, in a sunny site with a well-drained, moderately fertile soil; do not soak seed before planting and avoid excessive nitrogen, because bean is a legume that fixes its own nitrogen when properly inoculated and excess nitrogen reduces pod set. Space bush beans about 2-4 inches apart in rows 18-24 inches apart, and plant pole beans in hills or rows; install poles, trellises or other supports at planting time so that pole beans can climb as they grow, which also improves air movement and reduces disease. Water beans evenly, about an inch per week, and avoid overhead watering late in the day; mulch to conserve moisture and suppress weeds. Beans need pollination by bees for good pod set, so avoid insecticide applications during bloom. Scout weekly for bean beetles, aphids, spider mites and for the beetles and caterpillars that chew pods and leaves, and watch for the foliar diseases that follow wet weather. Harvest snap beans before the seeds inside the pods bulge, pick shelling beans when pods are thin and tough but not dry, and pick dry beans when pods are dry and the beans inside rattle. Regular picking keeps plants productive; store fresh beans in the refrigerator and handle pods gently because they bruise.",
            "https://extension.umn.edu/garden-and-home/yard-and-garden/gardening-in-minnesota/growing-beans",
            "University of Minnesota Extension (CFANS)",
            "2023-01-01", "north_america", "extension",
            ["bean", "legume", "spacing", "trellis", "harvest", "pollination"],
        ),

        # ---------- BELL PEPPER ----------
        "bell_pepper": Src(
            "University of California Statewide IPM Program (University of California Agriculture and Natural Resources)",
            "Cultural Tips for Growing Peppers",
            "Peppers are warm-season fruiting vegetables and require a long, warm growing season to produce mature fruit. UC IPM advises planting in full sun on a well-drained, fertile soil, setting transplants only after the soil and air have warmed and all frost danger has passed, because peppers are more sensitive to cold than tomatoes and set poorly in cool weather. Space plants about 12-18 inches apart in rows 24-36 inches apart. Water deeply but infrequently, soaking the soil rather than the foliage, and maintain even moisture during flowering and fruit set; use drip or furrow irrigation to keep leaves dry, and mulch to conserve moisture and reduce blossom-end rot, which is associated with irregular water supply and calcium transport problems rather than simple soil calcium shortage. Work compost and a balanced fertilizer into the bed before planting and side-dress or fertigate lightly after the first fruit sets; excessive nitrogen produces lush foliage and delayed fruit set. Peppers are self-pollinating, but a light buzz from bees and wind movement improves set, and fruit set is poor when temperatures are above about 90 F or below about 60 F. Scout weekly for aphids, thrips, pepper weevil and caterpillars, and for bacterial spot and other foliar diseases on leaves and fruit. Harvest green fruit when it reaches full size and is firm and glossy, or leave fruit to ripen to red, yellow or orange for higher sugar and vitamin content; cut fruit from the plant with the stem attached, and store at about 45-55 F and high humidity rather than at refrigerator temperatures that damage the fruit.",
            "https://ipm.ucanr.edu/home-and-landscape/cultural-tips-for-growing-peppers/",
            "University of California Statewide IPM Program (UC ANR)",
            "2023-01-01", "north_america", "extension",
            ["bell-pepper", "transplant", "blossom-end-rot", "fruit-set", "harvest", "uc-anr"],
        ),

        # ---------- BLUEBERRY ----------
        "blueberry": Src(
            "Penn State Extension (College of Agricultural Sciences)",
            "Blueberries in the Garden",
            "Blueberry is a long-lived perennial shrub that demands an acid, well-drained, high-organic-matter soil and will not thrive in ordinary garden conditions; Penn State Extension calls siting and soil preparation the most critical aspects of success. Take a soil test and amend the soil in the early autumn ahead of spring planting, aiming for an ideal pH of 4.5-5.0, and note that sulfur is added as iron or ammonium sulfate and not as aluminum sulfate, which is toxic to blueberries; because of these exacting requirements many gardeners build raised beds or use containers. A site where rhododendrons and azaleas thrive is a suitable site for blueberries: sunny with moist, porous, acidic soil. Plant before the soil warms and mulch with about four inches of organic matter such as aged sawdust, bark mulch or pine straw, keeping mulch clear of the crown and topping it up over time, because the shallow, fine roots dry out and die quickly and benefit from trickle irrigation during dry spells. Highbush blueberries can grow 5-8 feet tall and wide at maturity, so space them accordingly and allow for their full size; although they are largely self-pollinating, research shows that planting more than one cultivar produces larger berries and better crops, and selecting cultivars by ripening season can extend harvest from late June through September. Fertilize lightly in the establishment year — about a half-tablespoon of ammonium sulfate per plant in a circle about six inches from the stem, repeated about six weeks later — and top up nitrogen as needed rather than applying heavy doses at once, since low nitrogen is the deficiency most commonly seen. Keep a follow-up eye on soil pH, which creeps upward over time and will likely need further sulfur. Scout through the season for mummy berry, canker diseases and spotted-wing drosophila. Fruit turns blue three to four days before it is at its sweetest, so wait until berries are fully ripe before harvesting, because they will not ripen off the plant, and net against birds, which can strip a crop in a day; refrigerate or freeze promptly.",
            "https://extension.psu.edu/blueberries-in-the-garden-and-the-kitchen/",
            "Penn State Extension (College of Agricultural Sciences)",
            "2023-02-15", "north_america", "extension",
            ["blueberry", "acid-soil", "cross-pollination", "pruning", "harvest", "extension"],
        ),

        # ---------- BROCCOLI ----------
        "broccoli": Src(
            "University of California Statewide IPM Program (University of California Agriculture and Natural Resources)",
            "Cultural Tips for Growing Broccoli",
            "Broccoli is a cool-season cole crop that produces its best heads in cool weather and turns bitter or bolts when it matures in heat. UC IPM advises planting in full sun on a well-drained, fertile soil well supplied with organic matter, and setting transplants or sowing seed so that the crop matures before hot weather; in mild coastal areas broccoli can be grown almost year-round, while inland plantings are best in spring and autumn. Space plants about 12-18 inches apart in rows 24-36 inches apart, and water regularly to keep the soil evenly moist, because moisture stress reduces head size and quality; drip or furrow irrigation keeps foliage dry and reduces disease. Broccoli has a high nitrogen requirement and responds to a fertile soil with added compost or a balanced fertilizer before planting plus one or two side-dressings during growth; adequate phosphorus and potassium from a soil test support strong head development, and boron should be adequate because its shortage causes hollow stem. Rotate cole crops away from other brassicas to reduce soil-borne disease including clubroot, and maintain pH near 6.0-6.8, which both improves nutrient availability and suppresses clubroot. Scout weekly for cabbage aphid, cabbageworm caterpillars, flea beetles and diamondback moth, checking the growing point and the undersides of leaves where aphids cluster. Harvest the central head while buds are tight, deep green and about 4-8 inches across and before any yellow flowers open, cutting with several inches of stalk; side shoots then produce smaller secondary heads. Cool harvested heads quickly after cutting and store near 32 F and high humidity to preserve quality and vitamin content.",
            "https://ipm.ucanr.edu/home-and-landscape/cultural-tips-for-growing-broccoli/",
            "University of California Statewide IPM Program (UC ANR)",
            "2023-01-01", "north_america", "extension",
            ["broccoli", "cool-season", "spacing", "nitrogen", "harvest", "uc-anr"],
        ),

        # ---------- CABBAGE ----------
        "cabbage": Src(
            "University of California Statewide IPM Program (University of California Agriculture and Natural Resources)",
            "Cultural Tips for Growing Cabbage",
            "Cabbage is a hardy cool-season cole crop that will tolerate light frost but produces the best, most solid heads in cool, evenly moist conditions. UC IPM advises selecting a sunny, well-drained site with a fertile soil high in organic matter, and setting transplants in spring or autumn so the crop matures before or after the heat of summer; cabbage can also be overwintered in mild climates. Space plants about 12-18 inches apart in rows 24-36 inches apart to allow full head development and good air movement. Water regularly and evenly, because irregular moisture causes heads to split and reduces quality; drip or furrow irrigation keeps foliage dry. Cabbage is a heavy nitrogen user: work generous compost or a balanced fertilizer into the bed before planting and side-dress through the season, with adequate phosphorus and potassium from a soil test. Keep pH near 6.0-6.8, which both maximises nutrient availability and helps suppress clubroot, and rotate brassicas to avoid build-up of soil-borne pathogens and nematodes. Scout weekly for cabbageworm and other caterpillars, aphids, flea beetles and cabbage root maggot, inspecting the heads and the undersides of the wrapper leaves. Harvest when heads are firm and solid and reach a usable size, cutting the head at ground level; harvesting before the head splits is important because mature heads split quickly after rain. Withstand light frost but harvest before hard freeze. Store heads at about 32 F and high humidity for several months, and know that cabbage may release odours during storage.",
            "https://ipm.ucanr.edu/home-and-landscape/cultural-tips-for-growing-cabbage/",
            "University of California Statewide IPM Program (UC ANR)",
            "2023-01-01", "north_america", "extension",
            ["cabbage", "cole-crop", "head-splitting", "clubroot", "harvest", "uc-anr"],
        ),

        # ---------- CARROT ----------
        "carrot": Src(
            "University of California Statewide IPM Program (University of California Agriculture and Natural Resources)",
            "Cultural Tips for Growing Carrots",
            "Carrots are cool-season root crops that need a deep, loose, stone-free soil to produce long, straight roots; heavy or compacted ground causes forked and stunted carrots. UC IPM advises preparing the bed to a fine tilth by double-digging and working in organic matter, and sowing seed directly where the crop is to grow, because carrots transplant poorly. Sow seed in early spring or late summer for autumn harvest, planting thinly and covering lightly, and keep the seedbed moist with frequent light watering until seedlings emerge; carrots germinate slowly and a crusted surface prevents emergence. Thin seedlings to stand about 2-4 inches apart in rows 12-18 inches apart, because crowded carrots remain small and misshapen. Maintain even moisture through the season, because fluctuating water causes roots to split, and keep the soil surface covered with a light mulch to conserve moisture and moderate soil temperature. Carrots benefit from a soil with adequate potassium for root quality and adequate but not excessive nitrogen, since too much nitrogen produces large tops and small roots and can cause branching; avoid fresh manure, which can also cause forking and hairy roots. Scout through the season for carrot rust fly and for foliar disease, and watch for the dark, sunken boron-deficiency disorder in the root on sandy soils. Harvest when roots reach the desired size, pulling a test root to check, and harvest before the roots become over-mature and woody or split; a light frost improves flavour. Twist off the tops immediately, leaving a short stub, and store roots near 32 F and high humidity, since warm storage rapidly reduces quality.",
            "https://ipm.ucanr.edu/home-and-landscape/cultural-tips-for-growing-carrot/",
            "University of California Statewide IPM Program (UC ANR)",
            "2023-01-01", "north_america", "extension",
            ["carrot", "root-crop", "thinning", "soil-tilth", "harvest", "uc-anr"],
        ),

        # ---------- CAULIFLOWER ----------
        "cauliflower": Src(
            "University of California Statewide IPM Program (University of California Agriculture and Natural Resources)",
            "Cultural Tips for Growing Cauliflower",
            "Cauliflower is the least tolerant of the cole crops and needs a long, cool, evenly moist growing season; plants that are stressed by heat, drought or cold produce small, poor-quality curds. UC IPM advises a sunny site with a well-drained, fertile soil high in organic matter, and timing plantings so that curd development occurs in cool weather; in mild coastal areas cauliflower can be grown over a long season. Space transplants about 18-24 inches apart in rows 24-36 inches apart, and water regularly and evenly, since moisture stress at curd initiation causes buttoning and irregular supply causes poor curd development. Cauliflower has a high nitrogen requirement: incorporate compost and a balanced fertilizer before planting and side-dress during active growth to maintain steady supply, with adequate phosphorus and potassium from a soil test. Keep pH near 6.0-6.8 to maximise nutrient availability and reduce the molybdenum deficiency called whiptail that occurs on acid soils, and ensure boron is adequate because its shortage causes hollow stem and brown curd. Rotate brassicas to reduce soil-borne disease including clubroot. Scout weekly for cabbageworm, aphids, flea beetles and diamondback moth on the growing point and leaf undersides. Blanch the curd by tying the outer leaves over the head once the curd is about 2-3 inches across, protecting it from sunlight so the curd remains white; harvest when the curd is compact and firm and before it begins to loosen and rice out. Cool the heads promptly and store near 32 F and high humidity.",
            "https://ipm.ucanr.edu/home-and-landscape/cultural-tips-for-growing-cauliflower/",
            "University of California Statewide IPM Program (UC ANR)",
            "2023-01-01", "north_america", "extension",
            ["cauliflower", "blanching", "curd", "boron", "harvest", "uc-anr"],
        ),

        # ---------- CELERY ----------
        "celery": Src(
            "UF/IFAS Extension (University of Florida, Electronic Data Information Source)",
            "Celery and Cool-Season Vegetable Crop Culture in the Florida Garden",
            "Celery is a cool-season vegetable with a long growing season, a shallow root system and a high, continuous demand for water and nutrients, and the Florida Vegetable Gardening Guide published by UF/IFAS Extension treats it as a cool-weather crop in the home garden. Select a sunny site and prepare the soil well before planting, working in organic matter and adjusting pH to the range best for vegetable gardens on sandy soil, generally 5.8-6.3; have the soil tested so that lime, if needed, is applied two to three months before planting and thoroughly mixed to a depth of six to eight inches. Celery is normally grown from transplants rather than direct-seeded seed, and transplants should be healthy, free of insects and disease symptoms and not already flowering when they are set out. Set celery so that the rows are not crowded, because good air movement and even spacing reduce the foliar disease to which this shallow-rooted crop is prone; the guide advises rotating vegetables so that the same crop or plant family is not grown repeatedly in the same ground, and turning the soil well in advance of planting to discourage soil insects. Because the root system is shallow, celery cannot forage widely: keep the soil constantly and evenly moist and water deeply enough to wet the root zone, since both water and nutrient shortages cause small, pithy stalks and physiological disorders, and irregular moisture is associated with blackheart. Control weeds around and within the garden by mulching and hand-pulling, because weeds harbour the insects and diseases that attack celery, and choose adapted varieties with resistance or tolerance to nematodes and common diseases. Scout the crop through the season for the insects and diseases that thrive in hot weather, noting that late-summer plantings are most exposed to these problems. Harvest when the stalks are crisp and reach usable size, cutting the plant at the base, and keep harvested celery cool, since quality declines quickly in warm conditions.",
            "https://edis.ifas.ufl.edu/publication/VH021",
            "UF/IFAS Extension (University of Florida)",
            "2023-04-20", "north_america", "extension",
            ["celery", "cool-season", "transplant", "irrigation", "blackheart", "uf-ifas"],
        ),

        # ---------- CHERRY ----------
        "cherry": Src(
            "Penn State Extension (College of Agricultural Sciences)",
            "Fruit Production and Harvesting — Tree and Small Fruit",
            "Cherries are stone fruit that repay careful site selection and planting, because the trees are long-lived and early to bloom. Penn State Extension's fruit production guidance stresses that the first step is finding the right spot: most fruit crops, including cherry, require plenty of sunlight through the day to fuel fruit production, so choose a planting area that receives full sun for the majority of the day. Plant in early spring, and before planting make sure the soil is dry enough to crumble in your hand, since working wet soil damages its structure. Home orchards are advised to use dwarfing rootstocks, which keep trees manageable for pruning, spraying and harvest and bring the tree into bearing sooner. Penn State Extension notes that most tree fruits, and sweet cherries especially, set heavier crops when a second, compatible cultivar is planted nearby to supply pollen, so plan for pollination when choosing varieties, and give pollinisers and pollinators room for bloom-time bee activity. Because cherry blooms early, avoid frost pockets and be prepared to protect blossoms from late frost. Care through the season centres on maintaining an open canopy for air movement and light, watering during dry spells, and prompt removal of mummified fruit and infected wood. Ripening periods vary with variety, and Penn State Extension advises checking a few individual fruits for ripeness rather than judging by the calendar; access its resources on fruit maturity indicators and post-harvest handling to time harvest correctly. Handle harvested fruit gently and chill promptly, since tree fruit is perishable and quality falls quickly with rough handling or warm storage.",
            "https://extension.psu.edu/forage-and-food-crops/fruit/production-and-harvesting",
            "Penn State Extension (College of Agricultural Sciences)",
            "2023-04-10", "north_america", "extension",
            ["cherry", "polliniser", "brown-rot", "fruit-cracking", "harvest", "extension"],
        ),

        # ---------- CITRUS ----------
        "citrus": Src(
            "University of California Statewide IPM Program (University of California Agriculture and Natural Resources)",
            "Cultural Tips for Growing Citrus",
            "Citrus is a long-lived evergreen subtropical tree that needs full sun, a well-drained soil and protection from freezing temperatures. UC IPM advises planting in spring in a site with good air drainage where frost is least likely, setting the tree so that the graft union is well above the soil line and the root flare is exposed, and spacing trees according to rootstock and species — typically about 12-15 feet apart for standard trees. Citrus grows on a broad range of soils but dislikes poor drainage and standing water, so improve heavy soils with organic matter and avoid planting in low spots. Water deeply and infrequently to encourage deep rooting, allowing the soil to dry slightly between irrigations, and reduce or withhold irrigation in late autumn and winter to help the tree harden against cold; in containers and sandy soils more frequent watering is needed. Fertilize with a citrus-specific or balanced fertilizer containing nitrogen plus zinc, iron and manganese and, on sandy soils, magnesium, applying small amounts frequently through the warm growing season and stopping in late autumn. Iron and zinc deficiencies are common on high-pH soils and cause interveinal chlorosis of new leaves, which is managed by correcting pH and using chelated iron or foliar micronutrients rather than by adding more macronutrients. Citrus is self-fruitful and requires no polleniser; bees are the main pollinators but most common cultivars set fruit with little cross-pollination. Prune lightly to remove dead, damaged and crossing wood and to maintain a strong framework, avoiding heavy pruning that removes fruiting wood. Scout through the season for aphids, scale, citrus leafminer, thrips and Asian citrus psyllid, and for the diseases these can vector. Harvest when fruit reaches full colour and flavour, testing a fruit before picking the crop; citrus fruit holds well on the tree for weeks after colouring, and harvested fruit stores at about 45-50 F.",
            "https://ipm.ucanr.edu/home-and-landscape/cultural-tips-for-growing-citrus/",
            "University of California Statewide IPM Program (UC ANR)",
            "2023-01-01", "north_america", "extension",
            ["citrus", "evergreen", "micronutrients", "frost", "harvest", "uc-anr"],
        ),

        # ---------- COFFEE ----------
        "coffee": Src(
            "Food and Agriculture Organization of the United Nations (FAO)",
            "Coffee Cultivation and Crop Management (FAO Ecocrop / Good Agricultural Practices)",
            "Coffee is a perennial evergreen shrub or small tree of tropical highlands and is grown over a wide range of smallholder and estate systems. FAO guidance stresses that coffee performs best on deep, well-drained, fertile soils with good organic matter, usually of volcanic origin, and at altitudes where temperatures are moderate rather than extreme, since coffee is damaged by frost and by prolonged heat and drought. Propagation is by seed planted in shaded nurseries and transplanted as young seedlings at the onset of the rainy season, with shade trees or shade cloth used to protect young plants and, in many systems, to moderate temperatures and reduce stress on mature plants; planting density and spacing vary with species, cultivar and whether trees are kept as single-stem or multi-stem systems. Pruning is central to coffee management: it removes dead and unproductive wood, controls tree height for ease of harvest, regulates the number of bearing nodes and, in some systems, rejuvenates trees through stumping or by allowing suckers to replace old stems, which maintains yields over the life of the plantation. Weed control, mulching and, where necessary, terracing on slopes protect the soil from erosion and conserve moisture. Nutrient supply should follow soil and leaf analysis, with nitrogen, phosphorus, potassium, magnesium and boron the elements most often limiting, applied in split applications timed to the rains so that nutrients reach the root zone. Scout through the season for the major pests and diseases of the region, including coffee berry borer, leaf rust, coffee leaf miner and root-knot nematodes. Harvest selectively when berries turn fully red, picking only ripe cherries to protect quality, and process promptly by the wet or dry method; green beans store best when dried to a safe moisture content and kept cool and dry.",
            "https://www.fao.org/4/x6939e/x6939e00.htm",
            "Food and Agriculture Organization of the United Nations (FAO)",
            "2015-11-04", "global", "government",
            ["coffee", "perennial", "shade", "pruning", "selective-harvest", "fao"],
        ),

        # ---------- CORN ----------
        "corn": Src(
            "University of California Statewide IPM Program (University of California Agriculture and Natural Resources)",
            "Cultural Tips for Growing Corn",
            "Corn is a warm-season grass crop and a heavy feeder that grows best in full sun on a well-drained, fertile soil. UC IPM advises preparing the soil with compost and a balanced fertilizer before planting and sowing seed directly once the soil has warmed, since corn germinates poorly in cold, wet soil; plant in blocks of several short rows rather than a single long row to ensure good wind pollination, because corn is a wind-pollinated crop and poor pollination produces ears with missing kernels. Space plants about 8-12 inches apart in rows 24-36 inches apart, and thin to a uniform stand. Corn has a high nitrogen requirement: side-dress with nitrogen when plants are about a foot tall and again as tassels appear, because nitrogen shortage during ear development reduces yield and kernel fill; inadequate water at silking and ear fill also causes poor kernel development, so maintain even moisture and water deeply during flowering and ear growth. Sweet corn should be isolated in time or space from field corn and from different sweet corn types, because cross-pollination reduces sweetness and eating quality. Scout weekly through the season for corn earworm, European corn borer, fall armyworm and aphids, checking the whorl and developing ears, and manage weeds early because corn is a poor competitor when young. Harvest sweet corn when the silks are brown and dry and the kernels are plump and release a milky juice when pressed with a thumbnail; corn quality declines very rapidly after harvest because the sugar converts to starch, so pick just before use, chill immediately and refrigerate, or blanch and freeze within hours of picking for best quality.",
            "https://ipm.ucanr.edu/home-and-landscape/cultural-tips-for-growing-corn/",
            "University of California Statewide IPM Program (UC ANR)",
            "2023-01-01", "north_america", "extension",
            ["corn", "wind-pollination", "nitrogen", "earworm", "harvest", "uc-anr"],
        ),

        # ---------- CUCUMBER ----------
        "cucumber": Src(
            "University of Minnesota Extension (College of Food, Agricultural and Natural Resource Sciences)",
            "Growing Cucumbers in Home Gardens",
            "Cucumbers are warm-season vining crops that grow best in warm weather and are damaged by cold. UMN Extension advises starting seeds indoors in late April for transplants, or sowing seed directly in the garden after the soil has warmed, usually in May in Minnesota, and using plastic mulch and row covers to allow earlier planting and to protect plants from cold; cucumbers are among the most cold-sensitive of the common vegetables. Choose a sunny site with a well-drained, fertile soil high in organic matter, and space plants about 12 inches apart in rows 36-60 inches apart, or plant in hills. Cucumbers may be grown on the ground or trained up trellises and fences; vertical training saves space, keeps fruit clean and straight, improves air movement and makes harvest easier. Water regularly and deeply, about an inch per week, keeping moisture even because irregular watering produces bitter, misshapen fruit, and water at the base to keep foliage dry and reduce foliar disease. Cucumbers are heavy feeders: work compost and a balanced fertilizer into the bed before planting and side-dress lightly as vines begin to run, but avoid excessive nitrogen, which produces abundant vine growth and few fruit. Cucumbers bear separate male and female flowers on the same plant and depend on bees for pollination, so avoid insecticide applications during bloom and plant flowers nearby to attract pollinators; parthenocarpic (all-female) types grown under row covers set fruit without pollination. Scout regularly for cucumber beetles, which both feed on plants and transmit bacterial wilt, and for squash bugs, aphids and for powdery and downy mildew on the foliage. Harvest slicing cucumbers when they are 6-8 inches long, firm and uniformly green, and pickling types at 2-4 inches, harvesting every few days to keep vines productive; do not allow fruit to over-mature, and store cucumbers at about 50 F rather than at refrigerator temperatures, which cause chilling injury.",
            "https://extension.umn.edu/garden-and-home/yard-and-garden/gardening-in-minnesota/growing-cucumbers",
            "University of Minnesota Extension (CFANS)",
            "2023-01-01", "north_america", "extension",
            ["cucumber", "trellis", "pollination", "cucumber-beetle", "harvest", "umn"],
        ),
    }


# ---------- plant care records, part 2 (eggplant .. grapevine) ----------
def _plant_care_records_2() -> dict[str, Src]:
    return {
        # ---------- EGGPLANT ----------
        "eggplant": Src(
            "University of California Statewide IPM Program (University of California Agriculture and Natural Resources)",
            "Cultural Tips for Growing Eggplant",
            "Eggplant is a warm-season, long-season fruiting vegetable that needs more heat than tomato or pepper and grows poorly in cool conditions. UC IPM advises planting in full sun on a well-drained, fertile soil, setting transplants only after the soil has warmed thoroughly and all danger of frost has passed, and choosing the warmest part of the garden or using black plastic mulch and row covers to accumulate heat. Space plants about 18-24 inches apart in rows 30-36 inches apart, and stake or cage taller varieties to keep heavy-fruited plants upright, since fruit touching the ground is prone to rot. Water deeply and evenly through the season, since eggplant is intolerant of both drought and waterlogging and irregular moisture causes small, bitter fruit and blossom-end rot associated with calcium transport problems; drip irrigation keeps foliage dry and reduces foliar disease. Eggplant is a heavy feeder: incorporate compost and a balanced fertilizer before planting and side-dress or fertigate after the first fruit sets, without excess nitrogen, which delays fruiting. The flowers are self-pollinating but benefit from bee movement and wind, and fruit set fails in cool weather or when temperatures are excessively high. Scout weekly for flea beetles, Colorado potato beetle, aphids and spider mites, checking the undersides of leaves, and watch for verticillium wilt and for fruit rots in wet conditions. Harvest when the fruit is glossy, firm and full-sized but before the skin loses its shine and the flesh becomes seedy and bitter; cut fruit from the plant with a short stem rather than pulling. Harvested eggplant is perishable and sensitive to chilling injury, so store it at about 50-55 F and high humidity and use it within days rather than holding it at refrigerator temperature.",
            "https://ipm.ucanr.edu/home-and-landscape/cultural-tips-for-growing-eggplant/",
            "University of California Statewide IPM Program (UC ANR)",
            "2023-01-01", "north_america", "extension",
            ["eggplant", "heat-loving", "staking", "flea-beetle", "harvest", "uc-anr"],
        ),

        # ---------- GARLIC ----------
        "garlic": Src(
            "University of Minnesota Extension (College of Food, Agricultural and Natural Resource Sciences)",
            "Growing Garlic in Home Gardens",
            "Garlic is a cool-season bulb crop grown from cloves rather than seed, and UMN Extension advises planting in full sun on a well-drained soil that has been generously amended with compost or well-rotted manure, because garlic does not tolerate waterlogged ground and heavy clay soils should be improved or mounded before planting. In cold-winter climates plant hardneck garlic in autumn, about four to six weeks before the ground freezes so that cloves establish roots but do not produce top growth, setting cloves pointed end up about two inches deep and about four to six inches apart in rows 12 inches apart, and mulch heavily after planting to prevent winter heaving and to keep the soil cool and moist. Softneck garlic, which stores better and is usually the type grown in milder climates, can also be spring planted. Water regularly through spring and early summer while leaves and bulbs are developing, then stop watering as the plants begin to mature, since too much water late in the season causes bulbs to split and rot. Garlic is a modest feeder and generally needs little beyond the organic matter worked in at planting; avoid excess nitrogen, which produces lush top growth at the expense of bulb size and delays maturity. Keep the planting weed-free, because garlic is a poor competitor and because weeds harbour the diseases and insects that attack the crop. Scout through the season for onion thrips tearing the leaves and, in poorly drained ground, for the soil-borne rots that cause yellowing and collapse. Harvest when the lower leaves turn brown but about half the leaves are still green and the bulb is well formed, loosening the soil with a fork rather than pulling, and cure the bulbs in a warm, dry, airy place out of direct sun before trimming and storing; cured garlic stores for months in a cool, dry, dark location.",
            "https://extension.umn.edu/vegetables/growing-garlic",
            "University of Minnesota Extension (CFANS)",
            "2023-01-01", "north_america", "extension",
            ["garlic", "cloves", "autumn-planting", "curing", "thrips", "umn"],
        ),

        # ---------- GINGER ----------
        "ginger": Src(
            "Food and Agriculture Organization of the United Nations (FAO)",
            "Ginger Cultivation — Planting Material, Spacing and Crop Management (FAO)",
            "Ginger is a tropical, shade-tolerant rhizomatous crop grown from seed rhizomes rather than true seed, and FAO guidance on its cultivation notes that it requires a warm, humid climate with a long growing season and well-drained, fertile soils rich in organic matter; it is intolerant of waterlogging and of prolonged drought. Prepare the land deeply and thoroughly before planting, working in compost or well-rotted manure, and choose a site with partial shade where temperatures are high, since ginger performs best under light shade that moderates heat and moisture stress. Select sound, disease-free seed rhizomes with healthy buds and treat them before planting as recommended locally, then plant them in beds or on ridges at the start of the rainy season, setting pieces shallowly and covering with soil and a mulch layer; mulch is important because it conserves moisture, suppresses weeds and keeps soil temperatures moderate over the long season. Spacing varies with the production system but must allow the plants room to tiller and to develop rhizomes without crowding. Ginger has a shallow root system and depends on even moisture: irrigate during dry spells and ensure drainage in heavy rain, since waterlogging causes rhizome rots that can destroy the crop. Weed the crop carefully and by hand where necessary, because the developing rhizomes are shallow and easily damaged. Scout through the season for the bacterial wilt, rhizome rot and shoot borer that are the main constraints on the crop, and rotate away from ginger and related crops to reduce build-up of soil-borne disease. Harvest for use as fresh ginger once rhizomes reach usable size, and allow the crop to mature and foliage to yellow for the dried and processed product, lifting rhizomes carefully and curing them before storage.",
            "https://www.fao.org/4/t0207e/T0207E03.htm",
            "Food and Agriculture Organization of the United Nations (FAO)",
            "2015-11-04", "global", "government",
            ["ginger", "rhizome", "mulch", "drainage", "rot", "fao"],
        ),

        # ---------- GRAPE ----------
        "grape": Src(
            "Penn State Extension (College of Agricultural Sciences)",
            "Backyard Grape Growing",
            "Grapes are long-lived woody vines and are one of the most ancient crops known to humans, eaten fresh as table grapes or processed into juice, jelly, raisins and wine. Penn State Extension advises testing and amending the soil according to the soil test directions a year before planting, and notes that a well-established grapevine adapted to its climate will produce fruit for many decades, so the first year's goal is to establish the plant rather than to crop it. Plant vines in a hole a few inches deeper than the longest roots, trimming roots to 6-12 inches and soaking the vines in water before planting, and orient rows north-south to capture the most sunlight. While the vine is dormant, prune back to one or two canes leaving only two to three nodes per cane; after growth begins and frost danger has passed, remove all but the two strongest shoots, tie the shoots loosely to a training stake to build a straight trunk, and remove all flower clusters in the first year. Keep the new vines watered and weeded, and keep deer and rabbits away from the tender shoots. A standard commercial trellis is about 6 feet tall with supporting wires, and it can be erected in the first or second year, since the vine's form should be decided early because there is only one chance to train it correctly. Nutrition should follow a soil test: apply 2 ounces of 33-0-0 two to three weeks after planting, keeping fertilizer a foot from the vine, and in subsequent years 4, 6 or 8 ounces of 33-0-0 or 1-2 pounds of 10-10-10 per plant before buds swell in spring; if vines are too vigorous, omit nitrogen for one to two years, and maintain soil pH between 5.5 and 7.0. Prune in the dormant months between December and March, since pruning sets the bud number and therefore the crop; typically about 90 percent of a mature vine's new growth is removed, and grapes bear fruit on one-year-old wood. Monitor and control insect and disease pests, and harvest fruit when fully coloured and flavoured.",
            "https://extension.psu.edu/backyard-grape-growing/",
            "Barbara Goulart and Mark Chien, Penn State Extension (College of Agricultural Sciences)",
            "2023-05-11", "north_america", "extension",
            ["grape", "trellis", "pruning", "establishment", "harvest", "extension"],
        ),

        # ---------- GRAPEVINE ----------
        "grapevine": Src(
            "Penn State Extension (College of Agricultural Sciences)",
            "Backyard Grape Growing",
            "The productive life of a grapevine depends on training and pruning decisions made from the first year, because grapes bear their fruit on one-year-old wood and because the structure to which the vine is trained can be filled but must not be overgrown. Penn State Extension stresses that a mature grapevine naturally produces far more wood than it can support — much more than is necessary or desirable — and that typically about 90 percent of the new growth of a mature vine is removed during dormant pruning, leaving roughly three to four buds per foot of cordon. Pruning is carried out in the dormant months between December and March and sets the bud number and therefore the crop for the coming season: a vine that is too large carries more disease and produces lower-quality fruit, while a small vine is unproductive. Train vines so that one to two layers of leaves cover any area of the canopy, since that balance is best for flower-bud and fruit development, and note that different cultivars have different growth habits. Penn State Extension describes the vertical shoot position system used for vinifera and high-quality hybrid grapes, which creates a hedge-like wall of foliage with the fruit zone at the base; the fruiting wire is set 30-36 inches above the ground, green shoots are trained vertically and held with catch wires, and the vine is head-trained and cane-pruned or cordon-trained and spur-pruned. Vines can be supported on arbors, fences or standard trellises, but the training structure should be decided early because there is only one chance to train the vine correctly. Keep new vines watered and weeded, and control insect and disease pests throughout the season, since canopy management and pest control together determine fruit quality.",
            "https://extension.psu.edu/backyard-grape-growing/",
            "Barbara Goulart and Mark Chien, Penn State Extension (College of Agricultural Sciences)",
            "2023-05-11", "north_america", "extension",
            ["grapevine", "training", "pruning", "vertical-shoot-position", "bud-number", "extension"],
        ),
    }


# ---------- plant care records, part 3 (lettuce .. rice) ----------
def _plant_care_records_3() -> dict[str, Src]:
    return {
        # ---------- LETTUCE ----------
        "lettuce": Src(
            "University of California Statewide IPM Program (University of California Agriculture and Natural Resources)",
            "Cultural Tips for Growing Lettuce",
            "Lettuce is a cool-season leafy crop that grows quickly and is best suited to cool weather, turning bitter and bolting to seed when it matures in heat. UC IPM advises choosing a sunny site with a well-drained, fertile soil high in organic matter, and sowing seed or setting transplants so that the crop matures in cool conditions; lettuce can be grown almost year-round in mild coastal areas, and succession sowings every two to three weeks provide a continuous supply. Sow seed shallowly, because lettuce seed needs light to germinate, and thin or space plants so that leaves have room to develop — about 8-12 inches apart for head types and closer for leaf types. Keep the soil evenly moist at all times, because lettuce is shallow-rooted and quickly becomes bitter and tough under moisture stress; drip irrigation keeps the foliage dry and reduces the foliar diseases to which lettuce is prone. Lettuce is a light to moderate feeder: work compost into the bed before planting and side-dress lightly with nitrogen during growth, since excessive nitrogen makes the leaves soft and prone to disease and since too little produces pale, slow-growing plants. Boron and calcium should be adequate, and even moisture is important because irregular water supply and rapid growth are associated with tipburn, a physiological disorder of the leaf margins. Scout weekly for aphids, caterpillars, slugs and for the downy mildew, powdery mildew and sclerotinia that dominate lettuce disease problems, and remove affected leaves promptly. Harvest leaf lettuce by cutting outer leaves or the whole plant when leaves reach usable size, and head lettuce when the head is firm and full-sized, harvesting in the cool of the morning; lettuce wilts quickly and is perishable, so cool it immediately and store near 32 F and high humidity, washing just before use.",
            "https://ipm.ucanr.edu/home-and-landscape/cultural-tips-for-growing-lettuce/",
            "University of California Statewide IPM Program (UC ANR)",
            "2023-01-01", "north_america", "extension",
            ["lettuce", "cool-season", "succession-sowing", "tipburn", "harvest", "uc-anr"],
        ),

        # ---------- MAPLE ----------
        "maple": Src(
            "University of Minnesota Extension (College of Food, Agricultural and Natural Resource Sciences)",
            "Homemade Maple Syrup",
            "Maples are used for shade, ornamental and syrup production, and their care depends on the species and purpose. UMN Extension notes that four species of maple can be used for sap production in Minnesota, with sugar maple the popular choice for its sweet sap. For syrup production, complete tapping by mid-February in central and southern Minnesota and by the second week of March in the northern part of the state to catch the earliest sap runs, and tap only trees with a trunk diameter of at least 10 inches measured at 4 feet above the ground, with no more than two taps in any tree greater than 20 inches in diameter; for best sap production a tree should have a short trunk topped with abundant foliage, and the key to good grove management is cutting practices that favour the development and retention of such trees. Select a spot on the trunk 2-4 feet above the ground with sound wood, drill a hole 2 inches into the wood slanting slightly upward so sap can flow downward, lightly tap in the collection spout, and attach a bucket, plastic bag or tubing line, covering open buckets so rain and debris stay out. Sap does not flow every day: it runs when there is a rapid warming trend in early to mid-morning following a night when the temperature drops below freezing, and a single taphole normally produces between one quart and one gallon of sap per flow period, accumulating 10-12 gallons in a season. More broadly, plant and maintain maples with attention to site and soil, because many maples perform poorly on alkaline soil or in poorly drained, compacted ground; keep mulch clear of the trunk and protect the shallow root zone from disturbance.",
            "https://extension.umn.edu/gathering-wild-grown-plants-and-fungi/homemade-maple-syrup",
            "University of Minnesota Extension (CFANS)",
            "2023-01-01", "north_america", "extension",
            ["maple", "tapping", "sap", "sugar-maple", "grove-management", "umn"],
        ),

        # ---------- PEACH ----------
        "peach": Src(
            "Penn State Extension (College of Agricultural Sciences)",
            "Fruit Production and Harvesting — Tree Fruit",
            "Peaches and nectarines require a sunny site with deep, well-drained soil and good air movement, and because they bloom early they should not be planted in low, frosty pockets where blossoms can be killed. Penn State Extension's fruit production guidance stresses that the first step to a productive orchard is finding the right spot, since most fruit crops need plenty of sunlight through the day to fuel fruit production, and that orchard soil should be tested and amended well in advance of planting because established trees are difficult and slow to correct. Plant trees in early spring while dormant, and Penn State Extension advises using dwarfing rootstocks in home plantings, because rootstocks keep the tree to a manageable size for pruning, spraying and harvest and bring it into bearing sooner. Penn State Extension emphasises thinning and pruning as the core of crop-load management: trees often set more fruit than they can support, and leaving too much fruit weakens the tree and makes it more susceptible to pests, so thinning is described as a highly effective crop-load management tool for stone fruit and apple growers that improves fruit size, reduces the spread of disease and promotes return bloom the following season; pruning develops the desired tree shape, improves air circulation and increases produce quality. Ripening periods vary with cultivar, so check a few individual fruits for ripeness rather than judging by the calendar, and use the maturity indicators described by Penn State Extension to time harvest. Scout through the season for peach leaf curl, brown rot, bacterial spot and oriental fruit moth, and remove and destroy mummified fruit and cankered wood. Handle harvested fruit gently, since peaches bruise readily, and refrigerate promptly.",
            "https://extension.psu.edu/forage-and-food-crops/fruit/production-and-harvesting",
            "Jayson K. Harper, Ph.D., Tara Baugher, Ph.D., and Melanie Schupp, Penn State Extension",
            "2023-04-10", "north_america", "extension",
            ["peach", "thinning", "pruning", "rootstock", "maturity", "extension"],
        ),

        # ---------- PLUM ----------
        "plum": Src(
            "Penn State Extension (College of Agricultural Sciences)",
            "Fruit Production and Harvesting — Tree Fruit",
            "Plums and prunes need a sunny site with deep, well-drained soil and adequate space, and because they bloom early the site should avoid low, frosty pockets where blossoms can be killed. Penn State Extension's fruit production guidance stresses that site selection is the first and most important decision, because most fruit crops need plenty of sunlight through the day to produce well, and that soil should be tested and amended before planting since correction afterwards is slow and difficult. Plant dormant trees in early spring and, in home plantings, choose trees on dwarfing rootstocks where available, because rootstocks keep the tree to a manageable size for pruning, spraying and harvest and bring it into bearing sooner. Penn State Extension describes pruning and thinning as the central crop-management practices for tree fruit: fruit trees often set more fruit than they can support, and leaving too much fruit weakens the tree and makes it more susceptible to pests, so thinning improves fruit size, reduces disease spread and promotes return bloom, while pruning develops the desired tree shape, improves air circulation and increases produce quality. Because ripening periods differ among cultivars, Penn State Extension advises checking individual fruits for ripeness rather than relying on the calendar, and provides maturity indicators and post-harvest handling guidance to time harvest correctly. Scout through the season for black knot, brown rot, bacterial spot and plum curculio, and remove and destroy black knots and mummified fruit promptly. Handle harvested fruit gently and refrigerate promptly, since plums soften quickly after harvest.",
            "https://extension.psu.edu/forage-and-food-crops/fruit/production-and-harvesting",
            "Jayson K. Harper, Ph.D., Tara Baugher, Ph.D., and Melanie Schupp, Penn State Extension",
            "2023-04-10", "north_america", "extension",
            ["plum", "thinning", "pruning", "black-knot", "maturity", "extension"],
        ),

        # ---------- POTATO ----------
        "potato": Src(
            "University of California Statewide IPM Program (University of California Agriculture and Natural Resources)",
            "Cultural Tips for Growing Potatoes",
            "Potatoes are cool-season root crops grown from seed tubers, and they need a loose, well-drained soil rich in organic matter so that the developing tubers can size without distortion. UC IPM advises planting certified disease-free seed potatoes rather than supermarket potatoes, in full sun, in early spring for a summer crop or in late summer for an autumn crop; cut large tubers into pieces each bearing at least one or two eyes and allow the cut surfaces to suberise (dry and skin over) for a day or two before planting. Plant seed pieces about 4 inches deep and about 10-12 inches apart in rows 24-36 inches apart, and as the plants grow, hill soil up around the stems or grow the crop under a thick mulch, which keeps the tubers covered. Covering is essential because tubers exposed to light turn green and develop solanine, which is toxic and makes them unfit to eat. Water regularly and evenly through the season, since potatoes need steady moisture for good tuber size but are prone to rot in waterlogged soil; drip or furrow irrigation and good drainage are preferred. Potatoes are moderate feeders that benefit from a soil test: phosphorus and potassium applied before planting support tuber development, and nitrogen should be adequate early but not excessive, because too much nitrogen produces large tops and small tubers. Scout through the season for Colorado potato beetle, aphids, flea beetles and leafhoppers, and for the late blight and early blight that are the major foliar diseases of potato. Harvest new potatoes when the plants begin to flower, and harvest the main crop after the vines have died back and the skins have set, lifting carefully with a fork to avoid cutting tubers; cure the crop in a cool, dark, humid place for a week or two before storing at about 40 F in the dark, discarding any green, cut or rotting tubers.",
            "https://ipm.ucanr.edu/home-and-landscape/cultural-tips-for-growing-potato/",
            "University of California Statewide IPM Program (UC ANR)",
            "2023-01-01", "north_america", "extension",
            ["potato", "seed-tubers", "hilling", "greening", "harvest", "uc-anr"],
        ),

        # ---------- RASPBERRY ----------
        "raspberry": Src(
            "Penn State Extension (College of Agricultural Sciences) / Cornell Cooperative Extension",
            "Raspberry Production",
            "Raspberries are bramble fruits with perennial roots and biennial canes, and they require a sunny, well-drained site with good air movement because brambles are susceptible to root rots and to the cane and fruit diseases that follow wet, crowded conditions. Prepare the site thoroughly before planting, testing and amending the soil, and removing perennial weeds, because they are difficult to control once a row is established. Penn State Extension describes the two fruiting habits that determine pruning and harvest: field-grown summer-bearing raspberries produce their first significant crop in about the third year after planting, while primocane-bearing (fall-bearing) plants usually yield a significant crop in the second year. At maturity, about four years old, field plantings of red raspberries should produce roughly 5,000 pounds of fruit per acre, while black raspberry yields are about half those of red raspberries; high-tunnel production is at least double that of field production for both red and black raspberries, and tunnel-grown primocane-fruiting raspberries may produce a viable harvest the same year they are planted if set out early in the spring. Train and maintain the rows so that canes are supported and the planting is not overcrowded, which improves air movement and reduces fruit rots. Raspberry flowers are self-fruitful and pollinated by bees and wind, so avoid insecticide applications during bloom. Scout through the season for spotted-wing drosophila, cane borers and the cane blights, and remove and destroy infested or diseased canes. Harvest is critical to quality: Penn State Extension advises that raspberries must be firm, well coloured and free of insects and rot for market, and that if harvested at the proper time and handled carefully they remain in good condition for several days. Because the fruit is fragile it should be picked and packed directly into containers without further sorting, harvested at least once every three days and adjusted for weather, with losses to spotted-wing drosophila reduced if fruit is harvested daily or every other day. Cool berries promptly to remove field heat, and pick early in the day, since precooling before shipment significantly extends shelf life.",
            "https://extension.psu.edu/raspberry-production/",
            "Kathy Demchak, Senior Extension Associate, Penn State Extension (College of Agricultural Sciences)",
            "2021-12-09", "north_america", "extension",
            ["raspberry", "brambles", "primocane", "spotted-wing-drosophila", "postharvest", "harvest"],
        ),

        # ---------- RICE ----------
        "rice": Src(
            "University of Arkansas System Division of Agriculture, Cooperative Extension Service",
            "2025 Arkansas Rice Quick Facts",
            "Rice is grown in flooded or intermittently flooded paddies and its cultural management is organised around establishing a uniform stand and then managing water carefully through the season. Arkansas Cooperative Extension guidance sets out the key culture points. Seed germination requires moisture, oxygen and temperatures above 50 F, and emergence occurs in 5-28 days depending on the environment; plants progress through pre-tillering and tillering, then panicle initiation and differentiation, 50% heading, and finally grain fill to maturity, which takes 30-45 days and is reached at approximately 20% grain moisture. Ideally rice is planted when the soil reaches 60 F at 4 inches depth, with good seed-to-soil contact and seed placed 1/4 to 1 1/2 inches deep; optimum seeding dates run from about April 1 (central Arkansas) or April 10 (northern Arkansas), and drilling at roughly 30 seeds per square foot for varieties and 10 seeds per square foot for hybrids is typical, with seeding rate increased for broadcast or water-seeded methods and for poorer seedbeds. Recommended drill row widths are 4-10 inches, with 7.5 inches most common, and the desired final stand is 12-18 plants per square foot for varieties and 6-10 for hybrids, since stand uniformity matters as much as the count. Water management is central: apply the permanent flood around the 5th leaf or 1st tiller stage, use levee gates high enough to store rainfall but still prevent washouts, and design multiple-inlet rice irrigation for efficiency. Drain the field based on both time and maturity - 25-30 days past 50% heading (25 days for long grain, 30 for medium grain) and two-thirds straw-coloured kernels on silt loam or one-third on clay soils. Scout through the season for weeds, insects and diseases including the rice water weevil, rice stink bug and the stalk and panicle diseases common to the region, and manage nitrogen in two applications (preflood and at late boot) for hybrids or as a preflood or two-way split for varieties. Harvest when the grain has matured and dried, then dry promptly to safe storage moisture, because improper moisture management causes head rice loss and quality reduction.",
            "https://uaex.uada.edu/farm-ranch/crops-commercial-horticulture/rice/2025%20Arkansas%20Rice%20Quick%20Facts.pdf",
            "Jarrod Hardke, Rice Extension Agronomist, and Ralph Mazzanti, Program Associate, Rice Verification, University of Arkansas System Division of Agriculture",
            "2025-06-20", "north_america", "extension",
            ["rice", "seeding-dates", "flood-irrigation", "stand-count", "drainage", "arkansas-extension"],
        ),
    }


# ---------- plant care records, part 4 (soybean .. zucchini) ----------
def _plant_care_records_4() -> dict[str, Src]:
    return {
        # ---------- SOYBEAN ----------
        "soybean": Src(
            "University of Minnesota Extension (College of Food, Agricultural and Natural Resource Sciences)",
            "Soybean Growth Stages and Management",
            "Soybean is a warm-season legume grown for seed and is managed through clearly defined growth stages. UMN Extension explains that the soybean plant progresses from emergence through a vegetative phase, during which trifoliolate leaves are produced and nodes are formed on the main stem, into a reproductive phase that begins at flowering and continues through pod formation and seed fill to physiological maturity. Vegetative stages are named by the number of fully developed trifoliolate leaves on the main stem (V1, V2, and so on), while reproductive stages are designated R1 (beginning bloom) through R8 (full maturity). UMN Extension stresses that seeding at the right time and rate establishes the stand that carries the crop, and that soybeans should be planted when soil conditions allow into a firm, well-prepared seedbed; plant only high-quality, properly treated seed because soybean seed is sensitive to cold, wet soils and soil-borne disease. After establishment, cultural management focuses on planting soybean after the main flush of weeds in a well-planned rotation, using tillage and pre-emergence herbicides as needed to control weeds during the critical early period before the canopy closes, and scouting through the season for the insects, diseases and weeds that vary with the year. UMN Extension notes that maximum soybean yield is associated with the accumulation of 1,875 growing degree days before a killing frost. Scout fields at key stages, harvest when the grain has matured and the seed is at harvest moisture, and adjust the combine to minimise harvest losses, since properly timed harvest and machine adjustment protect both yield and seed quality.",
            "https://extension.umn.edu/growing-soybean/soybean-growth-stages",
            "University of Minnesota Extension (CFANS)",
            "2021-01-01", "north_america", "extension",
            ["soybean", "growth-stages", "seedbed", "rotation", "harvest", "umn"],
        ),

        # ---------- SQUASH ----------
        "squash": Src(
            "University of California Statewide IPM Program (University of California Agriculture and Natural Resources)",
            "Cultural Tips for Growing Squash",
            "Squash are warm-season vine crops grown for both summer (immature) and winter (mature) use, and they need warm weather and a sunny, well-drained site. UC IPM advises preparing a site with a well-drained, fertile soil high in organic matter, and planting after the soil has warmed, since squash are damaged by cold and set poorly in cool conditions; black plastic mulch and row covers help accumulate heat and protect young plants. Plant in hills or rows, spacing bush types about 2-3 feet apart and vining types 4-8 feet apart to allow for the mature spread of the vines, or train vining squash to a trellis where space is limited; a trellis also lifts the fruit off the ground, which reduces fruit rot and keeps fruit clean. Water regularly and deeply, about an inch per week, keeping moisture even because dry/wet cycles stress plants and can cause poor fruit set and quality; water at the base rather than over the foliage to reduce foliar disease. Squash are heavy feeders, so work compost and a balanced fertilizer into the bed before planting and side-dress lightly as the vines grow; excessive nitrogen produces abundant leaf and vine at the expense of fruit. Squash bear separate male and female flowers on the same plant and depend on bees for pollination, so avoid insecticide applications during bloom and plant flowers nearby to support pollinators; fruit set is poor in cool weather and when bee activity is low. Scout regularly for squash bug, squash vine borer, cucumber beetles and aphids, which also transmit disease, and for powdery and downy mildew on the leaves. For summer squash, harvest young fruit when it is tender and glossy and the skin can be nicked with a fingernail, picking every other day to keep plants producing; for winter squash, leave fruit on the vine until the rind is hard and the stem begins to shrivel, harvest with a short stem attached, and cure in a warm, dry place before storing at cool room temperature, since immature winter squash stores poorly.",
            "https://ipm.ucanr.edu/home-and-landscape/cultural-tips-for-growing-squash/",
            "University of California Statewide IPM Program (UC ANR)",
            "2023-01-01", "north_america", "extension",
            ["squash", "pollination", "squash-bug", "harvest", "curing", "uc-anr"],
        ),

        # ---------- STRAWBERRY ----------
        "strawberry": Src(
            "University of Minnesota Extension (College of Food, Agricultural and Natural Resource Sciences)",
            "Growing Strawberries in the Home Garden",
            "Strawberries are shallow-rooted, low-growing perennial plants and need full sun to produce maximum fruit — ten or more hours of sunlight a day is ideal, and they need at least six hours of direct sun — and they need a well-drained, fertile site, because they cannot tolerate wet feet or standing water. UMN Extension advises taking a soil test before planting to determine whether nutrients are needed, and working some well-rotted compost into the soil, which adds nutrients, improves drainage and increases microbial activity. Space plants 12 to 18 inches apart. Strawberries are self-fertile but require bees for pollination, and their flowers are vulnerable to frost, so protect the blossoms when frost is forecast. Set plants so that the crown sits at the soil surface, neither buried nor exposed, since planting too deeply causes crown rot and planting too shallowly exposes the roots to drying; dormant transplants may look dead on arrival but sprout quickly once planted, and fresh green growth appears within about a week of planting, so keep the plants moist and cool and plant them as soon as possible. Remove some of the runners throughout the season or the plants will take over the yard, and after removing flowers for a few weeks after planting, fruit can be picked later that same summer; one June-bearing plant can produce up to 120 new daughter plants in a single season. Varieties differ by ripening habit — June-bearing types produce fruit during a short window in late June to early July, while day-neutral types flower and fruit continuously when temperatures are moderate and typically produce from July through October — so choose varieties to suit the season and site. Water regularly through establishment, fruit development and after harvest, since the plants are shallow-rooted. Renovate June-bearing beds after harvest, thin plants within two weeks of harvest and apply compost to day-neutral plants if needed, then cover plants with straw mulch for overwintering in cold climates. Scout regularly for slugs, aphids, tarnished plant bug, spotted-wing drosophila and for the fruit rots and leaf diseases that follow wet conditions, and remove diseased foliage and overripe fruit. Harvest berries when they are fully coloured and firm, pick with the cap and a short stem attached, harvest frequently in warm weather, and cool berries promptly, since strawberries are very perishable.",
            "https://extension.umn.edu/gardening-minnesota/growing-strawberries-home-garden",
            "University of Minnesota Extension (CFANS)",
            "2023-01-01", "north_america", "extension",
            ["strawberry", "self-fertile", "runners", "frost-protection", "renovation", "harvest"],
        ),

        # ---------- TOBACCO ----------
        "tobacco": Src(
            "Penn State Extension (College of Agricultural Sciences)",
            "No-Till Innovations in Tobacco",
            "Tobacco is a warm-season specialty field crop that is traditionally transplanted by hand and then cultivated numerous times with machinery to eliminate weeds and competition and to loosen the soil. Penn State Extension describes this traditional production method on Lancaster County farms, where tobacco has been grown since colonial times and where about 7,000 acres produce the local Pennsylvania, or type 41, variety used for cigar wrappers. The traditional system depends on repeated cultivation: because the crop is transplanted rather than direct-seeded and because weeds must be controlled without the selective herbicides available for agronomic crops, growers plough and cultivate the soil, which leaves it loose and very apt to erode in heavy rain, so gullies and ditches can develop and valuable topsoil can be lost on sloping ground. Penn State Extension therefore promotes no-till transplanting for tobacco, developed with Amish and Mennonite growers, which plants the crop directly into a cover crop such as clover, rye or wheat and leaves a thick mat of residue protecting the soil; no-till increases soil health and reduces runoff, erosion and nutrient pollution, and growers who have adopted it report uniform, high-quality crops comparable to their no-till corn and alfalfa. Key culture points for tobacco are therefore careful site and soil management, timely transplanting in spring after frost risk has passed, weed control through cultivation or no-till cover-crop systems, and scouting for the insects and diseases that attack the foliage through the season. Harvest timing and method depend on the tobacco type (for example flue-cured, burley or dark air-cured), and harvested leaves are cured and graded carefully afterwards to preserve leaf quality through to market. Follow current, locally recommended variety, fertility, pest-management and curing practices, since these differ by tobacco type, soil and region.",
            "https://extension.psu.edu/no-till-innovations-in-tobacco-video/",
            "Jeffrey Graybill, Penn State Extension (College of Agricultural Sciences)",
            "2022-01-01", "north_america", "extension",
            ["tobacco", "transplanting", "cultivation", "no-till", "curing", "extension"],
        ),

        # ---------- TOMATO ----------
        "tomato": Src(
            "University of California Statewide IPM Program (University of California Agriculture and Natural Resources)",
            "Cultural Tips for Growing Tomato",
            "Tomatoes are warm-season fruiting vegetables and are the most commonly grown home-garden crop, but they require warmth and careful water management to perform well. UC IPM advises planting in full sun on a well-drained, fertile soil high in organic matter, and setting transplants only after the soil has warmed thoroughly and all danger of frost has passed, since tomatoes are injured by cold and set poorly in cool weather; in cool coastal areas choose early, cool-tolerant varieties. Space plants about 2-3 feet apart in rows 3-4 feet apart, and train or stake them: staking or caging keeps fruit and foliage off the ground, improves air movement, reduces fruit rots and makes harvest easier, while allowing plants to sprawl invites disease and fruit rot. Water deeply and evenly, wetting the soil rather than the foliage, and maintain consistent moisture because irregular supply — especially alternating drought and heavy water — is the main cause of blossom-end rot, a calcium-related disorder of the fruit; drip irrigation with mulch is the most reliable system for even moisture and dry foliage. Tomatoes are heavy feeders: work compost and a balanced fertilizer into the bed before planting and side-dress after the first fruit sets, avoiding excess nitrogen, which produces large vines and few fruit. Tomatoes self-pollinate within each flower, and fruit set is poor when day temperatures exceed about 90 F or night temperatures are very warm or very cool. Scout weekly for hornworms, fruitworms, aphids, whiteflies and spider mites, and for the fungal and bacterial diseases of foliage and fruit that dominate tomato production, removing affected leaves and fruit promptly. Harvest fruit when it is fully coloured and slightly soft, ripening fruit on the vine for best flavour, or pick at the breaker stage and ripen indoors at room temperature (not in the refrigerator, which damages flavour and texture) when birds or splitting are a problem; store mature fruit cool but not cold.",
            "https://ipm.ucanr.edu/home-and-landscape/cultural-tips-for-growing-tomato/",
            "University of California Statewide IPM Program (UC ANR)",
            "2023-01-01", "north_america", "extension",
            ["tomato", "staking", "blossom-end-rot", "fruit-set", "harvest", "uc-anr"],
        ),

        # ---------- ZUCCHINI ----------
        "zucchini": Src(
            "University of California Statewide IPM Program (University of California Agriculture and Natural Resources)",
            "Cultural Tips for Growing Squash",
            "Zucchini and other summer squash are warm-season vine crops harvested immature and repeatedly through the season, and they need warm weather and a sunny, well-drained site. UC IPM advises preparing a site with a well-drained, fertile soil high in organic matter, and planting after the soil has warmed, since squash are damaged by cold and set poorly in cool conditions; black plastic mulch and row covers help accumulate heat and protect young plants. Plant zucchini in hills or rows, spacing bush types about 2-3 feet apart to allow for their mature spread, or train plants to a trellis or tepee in smaller gardens, which also lifts fruit off the ground and reduces fruit rot. Water regularly and deeply, about an inch per week, keeping moisture even, and water at the base rather than over the foliage to reduce foliar disease; zucchini under stress produces fewer and poorer-quality fruit. Squash are heavy feeders, so work compost and a balanced fertilizer into the bed before planting and side-dress lightly as the plants grow, avoiding excess nitrogen, which produces abundant foliage at the expense of fruit. Zucchini bear separate male and female flowers on the same plant and depend on bees for pollination, so avoid insecticide applications during bloom and plant flowers nearby to support pollinators; fruit set is poor in cool weather and when bee activity is low, and unpollinated female flowers shrivel and drop. Scout regularly for squash bug, squash vine borer, cucumber beetles and aphids, which also transmit disease, and for powdery and downy mildew on the leaves. Harvest zucchini while the fruit is young, tender and glossy and the skin can be nicked with a fingernail — generally 6-8 inches long for zucchini — picking every other day to keep the plants producing, because fruit left to over-mature becomes seedy and tough and reduces further yield; the edible blossoms are also harvested on some types. Squash are perishable and are sensitive to chilling injury, so store harvested fruit cool but not at refrigerator temperatures that damage it, and use it within a few days.",
            "https://ipm.ucanr.edu/home-and-landscape/cultural-tips-for-growing-squash/",
            "University of California Statewide IPM Program (UC ANR)",
            "2023-01-01", "north_america", "extension",
            ["zucchini", "summer-squash", "pollination", "harvest", "squash-bug", "uc-anr"],
        ),
    }
