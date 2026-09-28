"""
Farmer-facing disease guides aligned with PhilRice guidance.

Primary reference:
  Major diseases of the rice plant in the Philippines (DA-PhilRice, 2023)
  https://www.philrice.gov.ph/wp-content/uploads/2023/02/Major-diseases-of-the-rice-plant-in-the-Philippines.pdf

Supplementary: PhilRice Crop Protection Division (dbmp.philrice.gov.ph) and
cultural-management advisories for the rainy season.
"""

from __future__ import annotations

import re
from copy import deepcopy
from dataclasses import dataclass, field

PHILRICE_SOURCE = (
    "DA-PhilRice — Major diseases of the rice plant in the Philippines (2023)"
)
PHILRICE_SOURCE_URL = (
    "https://www.philrice.gov.ph/wp-content/uploads/2023/02/"
    "Major-diseases-of-the-rice-plant-in-the-Philippines.pdf"
)

_FORM_RE = re.compile(
    r"\b(?:\d+(?:\.\d+)?%?\s*)?(WP|EC|SC|SL|SP|GR|WG|EW|L)\b",
    re.IGNORECASE,
)

_METHOD_LABELS = {
    "foliar": "Foliar spray",
    "granular": "Granular broadcast",
    "seed_soak": "Seed soak",
    "seed_dress": "Seed dressing",
}

_PACK_BY_FORM = {
    "WP": "wp",
    "EC": "ec",
    "SC": "sc",
    "SL": "sl",
    "SP": "wp",
    "GR": "gr",
    "WG": "wp",
    "EW": "ec",
    "L": "sl",
    "SEED": "seed",
}

_PACK_ICONS = {
    "wp": "bi-box-seam",
    "ec": "bi-droplet-half",
    "sc": "bi-eyedropper",
    "sl": "bi-droplet",
    "gr": "bi-grid-3x3-gap",
    "seed": "bi-seedling",
}

_PACK_COLORS = {
    "wp": "#2e7d32",
    "ec": "#e65100",
    "sc": "#0277bd",
    "sl": "#00695c",
    "gr": "#6a1b9a",
    "seed": "#5d4037",
}

_APPLY_BY_METHOD = {
    "foliar": {
        "how_to": [
            "Wear long sleeves, pants, rubber boots, gloves, a mask, and eye protection.",
            "Fill the knapsack halfway with clean water. Mix only what you will spray today.",
            "Add the measured product (WP/WG: make a slurry in a bucket first; EC/SC: measure with a dedicated cup).",
            "Fill the tank, shake, and spray both sides of the leaves until just wet — not dripping.",
            "Walk row by row with the wind at your back. Wash the sprayer, clothes, and hands after.",
        ],
        "wet_field": (
            "Lower standing water to about 2–3 cm so you can walk the rows. Spray the leaves, "
            "not the floodwater. Do not spray into deep flood or while rain is falling."
        ),
        "dry_field": (
            "Spray only when leaves are dry to the touch (no dew). Early morning (6–9 AM) or "
            "late afternoon (4–6 PM). Wait if rain is likely within 2–4 hours."
        ),
    },
    "granular": {
        "how_to": [
            "Wear gloves and boots. Do not handle granules with bare hands or inhale the dust.",
            "Weigh the hectare rate for your plot size (example: half a hectare uses half the kg/ha rate).",
            "Broadcast evenly by hand or granule applicator — do not dump in one corner of the field.",
            "Keep water in the field after broadcasting so granules dissolve and reach the plant base.",
        ],
        "wet_field": (
            "Best method: broadcast into 3–5 cm standing water. Do not drain for 3–5 days so the "
            "chemical stays in the paddy. Do not let treated water flow into fishponds or rivers."
        ),
        "dry_field": (
            "Apply on moist soil, then irrigate lightly if you can. Granules work poorly on "
            "bone-dry, cracked soil because they will not dissolve or move to the roots."
        ),
    },
    "seed_soak": {
        "how_to": [
            "This is a seed treatment — not a field spray. Use certified seed.",
            "Prepare the soak solution at the label (or PhilRice) rate in a clean basin.",
            "Soak seed for the time on the label (often 12–24 hours), then drain and air-dry in the shade.",
            "Sow in the nursery. Do not pour leftover soak water into rivers, canals, or fishponds.",
        ],
        "wet_field": "Not applied in the paddy. Treat seed before sowing, whether the field will be flooded later or not.",
        "dry_field": "Not applied in the paddy. Treat seed before sowing in wet or dry seedbeds.",
    },
    "seed_dress": {
        "how_to": [
            "This is a seed treatment — not a field spray. Wear a mask; avoid breathing powder.",
            "Place seed in a basin or plastic bag. Add the dressing rate (g per kg seed).",
            "Mix until every seed is evenly coated. Sow soon after dressing.",
            "Wash hands and tools. Keep treated seed away from food, children, and animals.",
        ],
        "wet_field": "Dress seed before sowing — do not broadcast this powder into standing water.",
        "dry_field": "Dress seed, then sow in the nursery or a dry direct-seeded field.",
    },
}


@dataclass
class PesticideRecommendation:
    """Registered crop-protection product (verify current FPA label before use)."""

    product: str
    active_ingredient: str
    timing: str
    rate: str
    note: str = ""
    form: str = "WP"
    method: str = "foliar"
    method_label: str = "Foliar spray"
    pack_key: str = "wp"
    pack_icon: str = "bi-box-seam"
    sign_color: str = "#1b4332"
    how_to: list[str] = field(default_factory=list)
    wet_field: str = ""
    dry_field: str = ""


# Farmer-facing how-to shown above the product cards on the result page.
APPLICATION_GUIDE = {
    "field_modes": [
        {
            "key": "wet",
            "title": "Wet / flooded field",
            "local_name": "Basa / may tubig",
            "summary": "Irrigated paddies with standing water.",
            "icon": "bi-water",
            "points": [
                "Granules (GR): broadcast into 3–5 cm standing water. Do not drain for 3–5 days.",
                "Foliar spray: lower water to 2–3 cm so you can walk. Spray the leaves — not the floodwater.",
                "Do not spray into deep flood. Deep water dilutes granules and washes spray off lower leaves.",
                "After granules, keep irrigation in the plot. Do not release treated water into fishponds or rivers.",
            ],
        },
        {
            "key": "dry",
            "title": "Dry / rainfed field",
            "local_name": "Tuyo / walang tubig",
            "summary": "Upland or rainfed rice with little or no standing water.",
            "icon": "bi-sun",
            "points": [
                "Foliar spray: apply only when leaves are dry (no dew or rain). Use a knapsack sprayer.",
                "Granules: spread on moist soil, then irrigate lightly if you can so granules reach the roots.",
                "Do not apply granules on bone-dry, cracked soil — they will not dissolve.",
                "If rain is likely within 2–4 hours, wait. Rain washes spray off dry-field plants quickly.",
            ],
        },
    ],
    "spray_steps": [
        {
            "title": "Protect yourself",
            "icon": "bi-person-check",
            "detail": (
                "Long sleeves, pants, rubber boots, gloves, mask, and goggles. "
                "Do not eat, drink, or smoke while mixing or spraying."
            ),
        },
        {
            "title": "Check weather and leaves",
            "icon": "bi-cloud-sun",
            "detail": (
                "Leaves must be dry to the touch. No rain for the next 2–4 hours. "
                "Avoid strong wind and midday heat. Best time: 6–9 AM or 4–6 PM."
            ),
        },
        {
            "title": "Set the field water",
            "icon": "bi-droplet-half",
            "detail": (
                "Flooded field: drain to 2–3 cm for foliar spray, or keep 3–5 cm for granules. "
                "Dry field: spray as is; granules need moist soil."
            ),
        },
        {
            "title": "Mix the correct rate",
            "icon": "bi-beaker",
            "detail": (
                "Use the product label. WP/WG: slurry first. EC/SC: dedicated measuring cup. "
                "Mix only what you will spray today."
            ),
        },
        {
            "title": "Spray or broadcast evenly",
            "icon": "bi-wind",
            "detail": (
                "Foliar: wet both sides of leaves; nozzle 30–40 cm from the canopy. "
                "Granular: broadcast evenly — do not dump in one spot."
            ),
        },
        {
            "title": "Clean up and wait for PHI",
            "icon": "bi-hourglass-split",
            "detail": (
                "Wash sprayer, clothes, and hands. Keep people and animals out until spray dries. "
                "Follow the pre-harvest interval (PHI) on the label before cutting rice."
            ),
        },
    ],
}

_METHOD_LABELS_FIL = {
    "foliar": "Spray sa dahon",
    "granular": "Granules na i-broadcast",
    "seed_soak": "Pagbabad ng binhi",
    "seed_dress": "Paglagay ng dressing sa binhi",
}

_APPLY_BY_METHOD_FIL = {
    "foliar": {
        "how_to": [
            "Magsuot ng mahabang manggas, pantalon, rubber boots, gloves, mask, at proteksyon sa mata.",
            "Punuin ng kalahati ang knapsack ng malinis na tubig. Haluin lamang ang gagamitin ngayong araw.",
            "Idagdag ang nasukat na produkto (WP/WG: gawing slurry muna sa timba; EC/SC: sukatin sa nakalaang tasa).",
            "Punuin ang tangke, iling, at i-spray ang magkabilang panig ng dahon hanggang basa — huwag patuluin.",
            "Maglakad kada hanay nang pabalik sa hangin. Hugasan ang sprayer, damit, at kamay pagkatapos.",
        ],
        "wet_field": (
            "Ibaba ang tubig sa palayan sa humigit-kumulang 2–3 cm para makalakad sa mga hanay. "
            "I-spray ang dahon, hindi ang baha. Huwag mag-spray sa malalim na baha o habang umuulan."
        ),
        "dry_field": (
            "Mag-spray lamang kapag tuyo ang dahon sa hipo (walang hamog). Madaling umaga (6–9 AM) o "
            "hapon (4–6 PM). Maghintay kung malamang umulan sa loob ng 2–4 oras."
        ),
    },
    "granular": {
        "how_to": [
            "Magsuot ng gloves at boots. Huwag hawakan ang granules nang walang proteksyon o sipsipin ang alikabok.",
            "Timbangin ang dosis kada ektarya ayon sa laki ng inyong lote (halimbawa: kalahating ektarya = kalahati ng kg/ha).",
            "I-broadcast nang pantay-pantay sa kamay o granule applicator — huwag ibuhos sa iisang sulok.",
            "Panatilihin ang tubig sa bukid pagkatapos mag-broadcast para matunaw ang granules at umabot sa puno.",
        ],
        "wet_field": (
            "Pinakamainam: i-broadcast sa 3–5 cm na nakatayong tubig. Huwag patuyuin sa loob ng 3–5 araw "
            "para manatili ang kemikal sa palayan. Huwag palabasin ang tinretong tubig sa palaisdaan o ilog."
        ),
        "dry_field": (
            "Ilagay sa basang lupa, tapos diligan nang bahagya kung maaari. Mahina ang epekto ng granules "
            "sa tuyo at bitak-bitak na lupa dahil hindi ito natutunaw o umaabot sa ugat."
        ),
    },
    "seed_soak": {
        "how_to": [
            "Paggamot ito sa binhi — hindi spray sa bukid. Gumamit ng certified seed.",
            "Ihanda ang solusyon ayon sa label (o PhilRice) sa malinis na palanggana.",
            "Ibabad ang binhi ayon sa oras sa label (madalas 12–24 oras), patuyuin, at paamuin sa lilim.",
            "Ihasik sa seedbed. Huwag ibuhos ang natirang tubig sa ilog, kanal, o palaisdaan.",
        ],
        "wet_field": "Hindi inilalagay sa palayan. Gamutin ang binhi bago maghasik, basahin man o hindi ang bukid mamaya.",
        "dry_field": "Hindi inilalagay sa palayan. Gamutin ang binhi bago maghasik sa basa o tuyong seedbed.",
    },
    "seed_dress": {
        "how_to": [
            "Paggamot ito sa binhi — hindi spray sa bukid. Magsuot ng mask; iwasan ang paglanghap ng pulbos.",
            "Ilagay ang binhi sa palanggana o plastic bag. Idagdag ang dosis (g kada kg ng binhi).",
            "Haluin hanggang pantay ang patong sa bawat butil. Ihasik agad pagkatapos mag-dress.",
            "Hugasan ang kamay at kagamitan. Itago ang tinretong binhi mula sa pagkain, bata, at hayop.",
        ],
        "wet_field": "I-dress ang binhi bago maghasik — huwag i-broadcast ang pulbos na ito sa nakatayong tubig.",
        "dry_field": "I-dress ang binhi, tapos ihasik sa seedbed o sa tuyong direct-seeded na bukid.",
    },
}

APPLICATION_GUIDE_FIL = {
    "field_modes": [
        {
            "key": "wet",
            "title": "Basa / binahang palayan",
            "local_name": "Basa / may tubig",
            "summary": "Patubig na palayan na may nakatayong tubig.",
            "icon": "bi-water",
            "points": [
                "Granules (GR): i-broadcast sa 3–5 cm na nakatayong tubig. Huwag patuyuin sa loob ng 3–5 araw.",
                "Foliar spray: ibaba ang tubig sa 2–3 cm para makalakad. I-spray ang dahon — hindi ang baha.",
                "Huwag mag-spray sa malalim na baha. Natutunaw ang granules at nahuhugasan ang spray sa ibabang dahon.",
                "Pagkatapos ng granules, panatilihin ang irigasyon. Huwag palabasin ang tinretong tubig sa palaisdaan o ilog.",
            ],
        },
        {
            "key": "dry",
            "title": "Tuyo / rainfed na palayan",
            "local_name": "Tuyo / walang tubig",
            "summary": "Upland o rainfed na palay na kaunti o walang nakatayong tubig.",
            "icon": "bi-sun",
            "points": [
                "Foliar spray: ilagay lamang kapag tuyo ang dahon (walang hamog o ulan). Gumamit ng knapsack sprayer.",
                "Granules: ikalat sa basang lupa, tapos diligan nang bahagya kung maaari para umabot sa ugat.",
                "Huwag maglagay ng granules sa tuyo at bitak-bitak na lupa — hindi ito matutunaw.",
                "Kung malamang umulan sa loob ng 2–4 oras, maghintay. Mabilis nahuhugasan ng ulan ang spray sa tuyong bukid.",
            ],
        },
    ],
    "spray_steps": [
        {
            "title": "Protektahan ang sarili",
            "icon": "bi-person-check",
            "detail": (
                "Mahabang manggas, pantalon, rubber boots, gloves, mask, at goggles. "
                "Huwag kumain, uminom, o manigarilyo habang naghahalo o nag-spray."
            ),
        },
        {
            "title": "Tingnan ang panahon at dahon",
            "icon": "bi-cloud-sun",
            "detail": (
                "Dapat tuyo ang dahon sa hipo. Walang ulan sa susunod na 2–4 oras. "
                "Iwasan ang malakas na hangin at init sa tanghali. Pinakamainam: 6–9 AM o 4–6 PM."
            ),
        },
        {
            "title": "Ayusin ang tubig sa bukid",
            "icon": "bi-droplet-half",
            "detail": (
                "Binahang palayan: patuyuin hanggang 2–3 cm para sa foliar spray, o panatilihing 3–5 cm para sa granules. "
                "Tuyong bukid: i-spray nang ganito; kailangan ng granules ang basang lupa."
            ),
        },
        {
            "title": "Haluin ang tamang dosis",
            "icon": "bi-beaker",
            "detail": (
                "Sundin ang label ng produkto. WP/WG: gawing slurry muna. EC/SC: gamitin ang nakalaang tasa. "
                "Haluin lamang ang gagamitin ngayong araw."
            ),
        },
        {
            "title": "I-spray o i-broadcast nang pantay",
            "icon": "bi-wind",
            "detail": (
                "Foliar: basain ang magkabilang panig ng dahon; nozzle 30–40 cm mula sa canopy. "
                "Granular: i-broadcast nang pantay — huwag ibuhos sa iisang lugar."
            ),
        },
        {
            "title": "Maglinis at hintayin ang PHI",
            "icon": "bi-hourglass-split",
            "detail": (
                "Hugasan ang sprayer, damit, at kamay. Iwasan ng tao at hayop hanggang matuyo ang spray. "
                "Sundin ang pre-harvest interval (PHI) sa label bago anihin ang palay."
            ),
        },
    ],
}

_METHOD_LABELS_ILO = {
    "foliar": "I-spray iti bulong",
    "granular": "Granules a i-broadcast",
    "seed_soak": "Panagbabaad ti bukel",
    "seed_dress": "Panangikabil ti dressing iti bukel",
}

_APPLY_BY_METHOD_ILO = {
    "foliar": {
        "how_to": [
            "Agsuot iti atiddog a manggas, pantalon, rubber boots, gloves, mask, ken proteksion iti mata.",
            "Punuen ti guddua ti knapsack iti nadalus a danum. Haluen laeng ti usaren ita nga aldaw.",
            "Inayon ti nasukat a produkto (WP/WG: aramiden a slurry nga umuna iti timba; EC/SC: sukaten iti naited a tasa).",
            "Punuen ti tangke, iyingying, ken i-spray ti dua a bangir ti bulong agingga a mabasa — saan a patuloyen.",
            "Magna kada aray nga agsubli iti angin. Buggoan ti sprayer, kawes, ken ima kalpasan.",
        ],
        "wet_field": (
            "Ibaba ti danum iti pagayan agingga iti agarup 2–3 cm tapno makapagna kadagiti aray. "
            "I-spray ti bulong, saan a ti baha. Saan nga ag-spray iti nauneg a baha wenno no agtudo."
        ),
        "dry_field": (
            "Ag-spray laeng no namaga ti bulong iti panagsagid (awan hamog). Nasapa a bigat (6–9 AM) wenno "
            "malem (4–6 PM). Aguray no mabalin nga agtudo iti uneg ti 2–4 oras."
        ),
    },
    "granular": {
        "how_to": [
            "Agsuot iti gloves ken boots. Saan a tignayen dagiti granules nga awan proteksion wenno agsipsip iti alikamen.",
            "Timbreken ti dosis kada ektarya segun iti kadakkel ti loteyo (kas pagarigan: guddua nga ektarya = guddua ti kg/ha).",
            "I-broadcast a napantay iti ima wenno granule applicator — saan nga ibukbok iti maymaysa a suli.",
            "Itultuloy ti danum iti talon kalpasan ti panag-broadcast tapno matunaw dagiti granules ken makadanon iti mula.",
        ],
        "wet_field": (
            "Kasayaatan: i-broadcast iti 3–5 cm nga agtaktakder a danum. Saan a patuyuen iti uneg ti 3–5 aldaw "
            "tapno agtalinaed ti kemikal iti pagayan. Saan a pagturongen ti naitratar a danum iti palaisdaan wenno karayan."
        ),
        "dry_field": (
            "Ikabil iti nabasa a daga, kalpasanna diligan bassit no mabalin. Nababa ti epekto dagiti granules "
            "iti namaga ken nabitak a daga gapu ta saanda a matunaw wenno makadanon iti ramut."
        ),
    },
    "seed_soak": {
        "how_to": [
            "Daytoy ket pananggagamod iti bukel — saan a spray iti talon. Usaren ti certified seed.",
            "Isagana ti solusion segun iti label (wenno PhilRice) iti nadalus a palanggana.",
            "Ibabad ti bukel segun iti oras iti label (masansan 12–24 oras), patuyuen, ken paamuen iti aniniwan.",
            "Ihasik iti seedbed. Saan nga ibukbok ti natda a danum iti karayan, kanal, wenno palaisdaan.",
        ],
        "wet_field": "Saan a maikabil iti pagayan. Gamoden ti bukel sakbay nga aghasik, mabasa man wenno saan ti talon kalpasanna.",
        "dry_field": "Saan a maikabil iti pagayan. Gamoden ti bukel sakbay nga aghasik iti nabasa wenno namaga a seedbed.",
    },
    "seed_dress": {
        "how_to": [
            "Daytoy ket pananggagamod iti bukel — saan a spray iti talon. Agsuot iti mask; liklikan ti panaganges iti pulbos.",
            "Ikabil ti bukel iti palanggana wenno plastic bag. Inayon ti dosis (g kada kg ti bukel).",
            "Haluen agingga a napantay ti banag iti tunggal butil. Ihasik a dagus kalpasan ti panag-dress.",
            "Buggoan ti ima ken ramit. Ittalem ti naitratar a bukel manipud iti makan, ubbing, ken animal.",
        ],
        "wet_field": "I-dress ti bukel sakbay nga aghasik — saan nga i-broadcast daytoy a pulbos iti agtaktakder a danum.",
        "dry_field": "I-dress ti bukel, kalpasanna ihasik iti seedbed wenno iti namaga a direct-seeded a talon.",
    },
}

APPLICATION_GUIDE_ILO = {
    "field_modes": [
        {
            "key": "wet",
            "title": "Nabasa / binaha a pagayan",
            "local_name": "Nabasa / adda danum",
            "summary": "Natubigan a pagayan nga adda agtaktakder a danum.",
            "icon": "bi-water",
            "points": [
                "Granules (GR): i-broadcast iti 3–5 cm nga agtaktakder a danum. Saan a patuyuen iti uneg ti 3–5 aldaw.",
                "Foliar spray: ibaba ti danum agingga iti 2–3 cm tapno makapagna. I-spray ti bulong — saan a ti baha.",
                "Saan nga ag-spray iti nauneg a baha. Matunaw dagiti granules ken mahugasan ti spray kadagiti baba a bulong.",
                "Kalpasan dagiti granules, itultuloy ti irigasion. Saan a pagturongen ti naitratar a danum iti palaisdaan wenno karayan.",
            ],
        },
        {
            "key": "dry",
            "title": "Namaga / rainfed a pagayan",
            "local_name": "Namaga / awan danum",
            "summary": "Upland wenno rainfed a pagay a bassit wenno awan ti agtaktakder a danum.",
            "icon": "bi-sun",
            "points": [
                "Foliar spray: ikabil laeng no namaga ti bulong (awan hamog wenno tudo). Usaren ti knapsack sprayer.",
                "Granules: iwaras iti nabasa a daga, kalpasanna diligan bassit no mabalin tapno makadanon iti ramut.",
                "Saan nga ikabil dagiti granules iti namaga ken nabitak a daga — saanda a matunaw.",
                "No mabalin nga agtudo iti uneg ti 2–4 oras, aguray. Napartak a mahugasan ti tudo ti spray iti namaga a talon.",
            ],
        },
    ],
    "spray_steps": [
        {
            "title": "Salakniban ti bagim",
            "icon": "bi-person-check",
            "detail": (
                "Atiddog a manggas, pantalon, rubber boots, gloves, mask, ken goggles. "
                "Saan a mangan, uminum, wenno agsigarilyo bayat a manghalo wenno ag-spray."
            ),
        },
        {
            "title": "Kitaen ti tiempo ken bulong",
            "icon": "bi-cloud-sun",
            "detail": (
                "Masapul a namaga ti bulong iti panagsagid. Awan tudo iti sumaruno a 2–4 oras. "
                "Liklikan ti napigsa nga angin ken pudot iti tengnga ti aldaw. Kasayaatan: 6–9 AM wenno 4–6 PM."
            ),
        },
        {
            "title": "Urnosen ti danum iti talon",
            "icon": "bi-droplet-half",
            "detail": (
                "Binaha a pagayan: patuyuen agingga iti 2–3 cm para iti foliar spray, wenno itultuloy ti 3–5 cm para kadagiti granules. "
                "Namaga a talon: i-spray a kastoy; masapul dagiti granules ti nabasa a daga."
            ),
        },
        {
            "title": "Haluen ti umiso a dosis",
            "icon": "bi-beaker",
            "detail": (
                "Suruten ti label ti produkto. WP/WG: aramiden a slurry nga umuna. EC/SC: usaren ti naited a tasa. "
                "Haluen laeng ti usaren ita nga aldaw."
            ),
        },
        {
            "title": "I-spray wenno i-broadcast a napantay",
            "icon": "bi-wind",
            "detail": (
                "Foliar: basaan ti dua a bangir ti bulong; nozzle 30–40 cm manipud iti canopy. "
                "Granular: i-broadcast a napantay — saan nga ibukbok iti maymaysa a lugar."
            ),
        },
        {
            "title": "Dalusan ken urayen ti PHI",
            "icon": "bi-hourglass-split",
            "detail": (
                "Buggoan ti sprayer, kawes, ken ima. Liklikan dagiti tao ken animal agingga a mamaga ti spray. "
                "Suruten ti pre-harvest interval (PHI) iti label sakbay nga agani iti pagay."
            ),
        },
    ],
}


@dataclass
class FarmerGuide:
    """Structured management guide shown on the scan result page."""

    key: str
    name: str
    pathogen: str
    disease_type: str  # fungal, bacterial, viral, pest
    local_names: list[str] = field(default_factory=list)
    summary: str = ""
    symptoms: list[str] = field(default_factory=list)
    procedure: list[str] = field(default_factory=list)
    cultural: list[str] = field(default_factory=list)
    chemical: list[str] = field(default_factory=list)
    pesticides: list[PesticideRecommendation] = field(default_factory=list)
    prevention: list[str] = field(default_factory=list)
    resistant_varieties: list[str] = field(default_factory=list)
    farmer_tip: str = ""

    PESTICIDE_DISCLAIMER = (
        "Use only FPA-registered products for rice. Follow the label, PHI (pre-harvest interval), "
        "and advice from your municipal agriculturist (MAO). Rates below are typical references — "
        "always use the label on your purchased product."
    )

    def to_recommendation_text(self) -> str:
        """Flat text stored on LeafScan and parsed by recommendation_display."""
        lines = [f"{self.name} ({self.pathogen}):"]
        for i, step in enumerate(self.procedure, 1):
            lines.append(f"{i}. {step}")
        return " ".join(lines)


def _guide(**kwargs) -> FarmerGuide:
    return FarmerGuide(**kwargs)


def _infer_form(product: str) -> str:
    match = _FORM_RE.search(product)
    if match:
        return match.group(1).upper()
    lowered = product.lower()
    if "bleach" in lowered or "seed" in lowered:
        return "SEED"
    return "WP"


_SEED_MARKERS = (
    "seed treat",
    "seed soak",
    "seed dress",
    "(seed",
    "for seed",
    "before sowing",
    "before nursery",
)


def _infer_method(product: str, timing: str, note: str, form: str) -> str:
    product_l = product.lower()
    combined = f"{product_l} {timing.lower()} {note.lower()}"
    is_seed = form == "SEED" or any(marker in combined for marker in _SEED_MARKERS)
    if is_seed:
        if "soak" in product_l or "bleach" in product_l:
            return "seed_soak"
        return "seed_dress"
    if form == "GR":
        return "granular"
    return "foliar"


def _p(product, active, timing, rate, note="") -> PesticideRecommendation:
    form = _infer_form(product)
    method = _infer_method(product, timing, note, form)
    apply = _APPLY_BY_METHOD[method]
    pack_key = "seed" if method in ("seed_soak", "seed_dress") else _PACK_BY_FORM.get(form, "wp")
    return PesticideRecommendation(
        product=product,
        active_ingredient=active,
        timing=timing,
        rate=rate,
        note=note,
        form=form,
        method=method,
        method_label=_METHOD_LABELS[method],
        pack_key=pack_key,
        pack_icon=_PACK_ICONS.get(pack_key, "bi-box-seam"),
        sign_color=_PACK_COLORS.get(pack_key, "#1b4332"),
        how_to=list(apply["how_to"]),
        wet_field=apply["wet_field"],
        dry_field=apply["dry_field"],
    )


# Shared fungicide sets (PhilRice CPD / common Philippine rice labels)
_BLAST_FUNGICIDES = [
    _p("Tricyclazole 75 WP", "Tricyclazole", "Panicle initiation (PI) + early heading", "300–500 g/ha in 200–400 L water", "Systemic; best for blast. 2 sprays 7–10 days apart if heavy pressure."),
    _p("Isoprothiolane 40 EC", "Isoprothiolane", "First lesions at tillering, repeat at PI", "1.0–1.5 L/ha", "Protectant + systemic; alternate with tricyclazole to delay resistance."),
    _p("Azoxystrobin 250 SC", "Azoxystrobin", "Tillering to booting when spots spread", "300–400 mL/ha", "Broad-spectrum; combine with good drainage and lower nitrogen."),
    _p("Propiconazole 250 EC", "Propiconazole", "Booting to heading (neck blast risk)", "0.5–1.0 L/ha", "Also used for brown spot and sheath diseases."),
]

_BROWN_SPOT_FUNGICIDES = [
    _p("Mancozeb 80 WP", "Mancozeb", "At first spots; repeat every 7–10 days if needed", "2.0–2.5 kg/ha in 200–400 L water", "Protectant; safe baseline fungicide for leaf spots."),
    _p("Propiconazole 250 EC", "Propiconazole", "Tillering to flowering when outbreak spreads", "0.5–1.0 L/ha", "Curative action; fix soil fertility too."),
    _p("Iprodione 50 WP", "Iprodione", "Active spreading lesions on leaves", "1.0–1.5 kg/ha", "Alternate with mancozeb; follow PHI on label."),
    _p("Carbendazim 50 WP", "Carbendazim", "Moderate infection before flowering", "500 g–1 kg/ha", "Do not rely on one chemical all season."),
]

GUIDES: dict[str, FarmerGuide] = {
    "bacterial blight": _guide(
        key="bacterial blight",
        name="Bacterial Leaf Blight",
        pathogen="Xanthomonas oryzae pv. oryzae",
        disease_type="bacterial",
        local_names=["BLB", "Kresek (seedling wilt)"],
        summary=(
            "A major bacterial disease in the Philippines, especially during the rainy season. "
            "Infected leaves dry from the tip downward and yield can drop sharply in susceptible fields."
        ),
        symptoms=[
            "Water-soaked streaks along leaf margins that turn yellow then white/gray",
            "Leaves wilt and dry from the tip (kresek) on young plants",
            "Milky bacterial ooze from cut leaf — place in clear water to check",
            "More severe with high nitrogen and standing water",
        ],
        procedure=[
            "Scout weekly during tillering and flowering; mark hot spots in the field map.",
            "Remove and burn or bury heavily infected plants and weed hosts near dikes.",
            "Drain the field briefly after heavy rain; avoid deep flooding that spreads bacteria.",
            "Split nitrogen application — avoid heavy late nitrogen that increases susceptibility.",
            "For the next crop, use BLB-resistant varieties (e.g. NSIC Rc 222, Rc 402) from certified seed.",
            "Plow under rice stubble after harvest; allow field to dry before the next planting.",
        ],
        cultural=[
            "Field sanitation: remove volunteer rice, ratoons, and weeds that host the bacteria.",
            "Shallow water in the nursery; good drainage during severe flooding.",
            "Proper plant spacing and balanced fertilisation (not excess nitrogen).",
            "Synchronise planting with neighbours to shorten the disease window.",
        ],
        chemical=[
            "PhilRice notes chemical control of BLB is often not economical or highly effective.",
            "Spray only when advised by MAO — resistant varieties and sanitation come first.",
        ],
        pesticides=[
            _p("Copper hydroxide 77 WP", "Copper hydroxide", "Early infection on leaf tips (optional)", "2–3 kg/ha in 400 L water", "Bactericide; repeat every 7–10 days max per label. Not a substitute for resistant varieties."),
            _p("Copper oxychloride 50 WP", "Copper oxychloride", "Same as above where registered for rice", "3–4 kg/ha", "Spray in late afternoon; avoid tank-mix unless label allows."),
            _p("Streptomycin sulfate + tetracycline", "Antibiotics", "Only if MAO/pathologist prescribes", "Per label (often 200 ppm spray)", "Check FPA registration; overuse causes resistance. PhilRice prefers prevention."),
            _p("Bleaching powder (seed soak)", "Calcium hypochlorite", "Seed treatment before sowing", "100 µg/mL soak per PhilRice", "Seed treatment, not field spray — reduces seed-borne bacteria."),
        ],
        prevention=[
            "Plant certified, disease-free seed from resistant varieties suited to your ecosystem.",
            "Avoid working in the field when leaves are wet (spreads bacteria).",
            "Do not move infected straw or seedlings to healthy fields.",
        ],
        resistant_varieties=[
            "Ask your municipal agriculturist for BLB-resistant varieties for your region "
            "(e.g. NSIC Rc 222, Rc 402 / Tubigan 36 where recommended).",
        ],
        farmer_tip=(
            "Prevention beats spraying: PhilRice emphasises resistant varieties, clean fields, "
            "and balanced nitrogen rather than routine antibiotic or copper sprays."
        ),
    ),
    "blast": _guide(
        key="blast",
        name="Rice Blast",
        pathogen="Pyricularia oryzae (Magnaporthe oryzae)",
        disease_type="fungal",
        local_names=["Mata-mata", "Agupaw", "Taya-taya"],
        summary=(
            "One of the most destructive rice diseases in the Philippines. Spores spread by wind "
            "and rain; lesions appear on leaves, nodes, and panicles."
        ),
        symptoms=[
            "Diamond- or spindle-shaped spots with gray centre and brown margin (leaf blast)",
            "Rotten black node that may break (node/collar blast)",
            "Blank or partially filled grains; panicle may not emerge (neck/panicle blast)",
            "Worse with cool nights, dew, dense canopy, and excess nitrogen",
        ],
        procedure=[
            "Inspect leaves at tillering and again at booting — treat early, not after widespread damage.",
            "Reduce nitrogen; apply in small split doses (ammonium forms are less conducive than excess nitrate).",
            "Improve drainage in rainfed areas to avoid drought stress followed by rewetting.",
            "Avoid farm work when plants are wet; increase spacing if the canopy is too dense.",
            "Burn or bury infected stubble after harvest; do not leave blast-infected straw on the field.",
            "If approved locally, apply fungicide at early heading and again when panicles emerge "
            "(e.g. tricyclazole, isoprothiolane — follow label and LGU advice).",
        ],
        cultural=[
            "Plant blast-resistant varieties (e.g. PSB Rc 10 or varieties recommended for your area).",
            "Transplant at proper spacing rather than dense broadcast; water-seeding can reduce some stress.",
            "Time planting so the crop does not face heavy dew during the most susceptible stage.",
        ],
        chemical=[
            "Apply fungicide when lesions spread and variety is susceptible — two sprays around heading.",
        ],
        pesticides=list(_BLAST_FUNGICIDES) + [
            _p("Kasugamycin 3% SL", "Kasugamycin", "Leaf blast at tillering (where registered)", "500 mL–1 L/ha", "Antibiotic fungicide; use with IPM, not routine calendar spray."),
            _p("Benomyl / carbendazim (legacy)", "Benzimidazole", "Only if still on local FPA list", "Per label", "PhilRice CPD also lists benomyl — confirm current registration."),
        ],
        prevention=[
            "Use disease-free seed and resistant varieties every season.",
            "Monitor neighbouring fields; blast spores travel long distances in humid weather.",
        ],
        resistant_varieties=["PSB Rc 10 and other NSIC varieties with blast resistance for your ecosystem."],
        farmer_tip="Neck blast at heading causes the largest yield loss — scout before panicle emergence.",
    ),
    "leaf blast": _guide(
        key="leaf blast",
        name="Leaf Blast",
        pathogen="Pyricularia oryzae",
        disease_type="fungal",
        local_names=["Mata-mata", "Dapog blast (seedling stage)"],
        summary="Leaf-stage blast common from seedling to tillering. Early control prevents spread to nodes and panicles.",
        symptoms=[
            "Small spindle-shaped lesions with gray centres on leaves",
            "Lesions may merge and kill leaf tissue",
            "Often appears first on lower leaves in humid, shaded parts of the field",
        ],
        procedure=[
            "Remove severely infected leaves or tillers when infection is still localised.",
            "Lower nitrogen rate; split applications instead of one heavy dose.",
            "Keep irrigation stable — avoid prolonged leaf wetness overnight.",
            "Plan fungicide only if lesions are spreading and variety is susceptible.",
            "Switch to blast-resistant variety in the next cropping season.",
        ],
        cultural=[
            "Proper spacing and weed control to improve air movement.",
            "Avoid late-evening irrigation that keeps leaves wet all night.",
        ],
        chemical=["Spray at first spreading lesions; repeat per label if rain washes chemicals off."],
        pesticides=_BLAST_FUNGICIDES[:3],
        prevention=["Resistant varieties and balanced fertilisation are the foundation of blast control."],
        farmer_tip="Leaf blast today can become neck blast at heading — act while lesions are still limited.",
    ),
    "neck blast": _guide(
        key="neck blast",
        name="Neck / Panicle Blast",
        pathogen="Pyricularia oryzae",
        disease_type="fungal",
        local_names=["Panicle blast", "Neck rot"],
        summary=(
            "Infection at the neck of the panicle causes empty grains and direct yield loss. "
            "Even 10% neck infection can cause measurable yield and grain-quality loss."
        ),
        symptoms=[
            "Brown to gray lesion girdling the panicle neck",
            "Panicle may break or remain inside the boot",
            "Partially filled or chalky grains",
        ],
        procedure=[
            "Apply recommended fungicide at panicle initiation and early heading if the variety is susceptible.",
            "Do not apply heavy nitrogen near heading.",
            "Mark affected areas for resistant variety replacement next season.",
            "Harvest promptly; bury or burn infected straw.",
        ],
        cultural=[
            "Plant early enough to avoid peak disease pressure during heading.",
            "Use resistant varieties in blast-prone irrigated and upland areas.",
        ],
        chemical=["Two fungicide sprays at booting and heading are critical for neck/panicle blast."],
        pesticides=[
            _p("Tricyclazole 75 WP", "Tricyclazole", "Booting + early heading (mandatory timing)", "300–500 g/ha each spray", "Most important spray window for yield protection."),
            _p("Propiconazole 250 EC", "Propiconazole", "Alternate spray if tricyclazole unavailable", "0.5–1.0 L/ha", "Apply before neck lesions girdle the panicle."),
            _p("Isoprothiolane 40 EC", "Isoprothiolane", "Booting stage backup", "1.0–1.5 L/ha", "Good when humidity is high at heading."),
        ],
        prevention=["Neck blast management starts with variety choice and nitrogen management weeks earlier."],
        farmer_tip="This stage causes the biggest harvest loss — prioritise resistant varieties in blast hotspots.",
    ),
    "brown spot": _guide(
        key="brown spot",
        name="Brown Spot",
        pathogen="Cochliobolus miyabeanus (Bipolaris oryzae)",
        disease_type="fungal",
        local_names=["Sesame leaf spot", "Helminthosporiosis"],
        summary=(
            "Common fungal leaf disease, often linked to poor soil fertility, drought stress, or weak plants. "
            "PhilRice lists it among major leaf diseases in the Philippines."
        ),
        symptoms=[
            "Small circular to oval brown spots with gray or tan centres on leaves",
            "Spots may enlarge and merge; severely infected leaves may dry",
            "Also seen on leaf sheaths and grains in bad outbreaks",
            "More common when plants lack balanced nutrients or suffer moisture stress",
        ],
        procedure=[
            "Confirm spots match brown spot (round lesions with gray centres) versus blast (diamond-shaped).",
            "Improve soil fertility — apply balanced NPK and organic matter as recommended by soil test.",
            "Correct silicon and potassium deficiency if your soil test shows low levels.",
            "Maintain shallow, even irrigation; avoid prolonged drought during tillering and flowering.",
            "Remove heavily infected crop residue after harvest; plow stubble under.",
            "If the outbreak is spreading fast, apply mancozeb or propiconazole per label and LGU guidance.",
            "Next season use certified, fungicide-treated seed from a resistant or tolerant variety.",
        ],
        cultural=[
            "Balanced fertilisation — avoid starving the crop then applying excess nitrogen late.",
            "Good water management; reduce plant stress during critical growth stages.",
            "Crop rotation and field sanitation where brown spot recurs every season.",
        ],
        chemical=[
            "Spray fungicide when spots spread after you have corrected fertility and water stress.",
        ],
        pesticides=list(_BROWN_SPOT_FUNGICIDES) + [
            _p("Azoxystrobin + difenoconazole SC", "Azoxystrobin / Difenoconazole", "Severe leaf spot during tillering", "300–500 mL/ha", "Broad-spectrum; use after mancozeb if needed."),
        ],
        prevention=[
            "Healthy, fertile soil is the best defence — brown spot often signals nutrient or moisture stress.",
            "Use quality seed and avoid saving grain from heavily infected fields.",
        ],
        resistant_varieties=["Ask PhilRice or your MAO for varieties with good brown-spot tolerance in your area."],
        farmer_tip=(
            "PhilRice stresses correcting soil fertility and water stress — fungicide alone will not fix "
            "a nutrient-weak field."
        ),
    ),
    "sheath blight": _guide(
        key="sheath blight",
        name="Sheath Blight",
        pathogen="Rhizoctonia solani",
        disease_type="fungal",
        local_names=["Awan-awan", "Banded leaf sheath blight"],
        summary=(
            "Fungal disease favoured by high humidity, dense canopy, and excess nitrogen. "
            "Lesions climb from leaf sheaths and can damage the panicle."
        ),
        symptoms=[
            "Oval or irregular greenish-gray lesions with brown margin on leaf sheaths",
            "Lesions can merge and rot the sheath; upper leaves may die",
            "Sclerotia (small dark pellets) form on infected tissue",
            "Worse in densely planted, heavily fertilised fields",
        ],
        procedure=[
            "Scout along field edges and low spots where humidity stays high.",
            "Reduce plant density next time — wider spacing improves air flow.",
            "Lower nitrogen; avoid lush, weak stems that favour infection.",
            "Drain the field periodically if water is standing too long.",
            "Apply validamycin, hexaconazole, or propiconazole when lesions first appear on sheaths.",
            "Do not use infected stubble as mulch.",
        ],
        cultural=[
            "Balanced fertilisation and recommended planting density.",
            "Periodic drainage in irrigated fields during the rainy season.",
        ],
        chemical=["Spray when lesions first appear on lower sheaths — not after the flag leaf is infected."],
        pesticides=[
            _p("Validamycin 3% L", "Validamycin", "First sheath lesions at mid-tillering", "1.5–2.5 L/ha in 400 L water", "Standard PhilRice-recommended bactericide for sheath blight."),
            _p("Hexaconazole 5 SC", "Hexaconazole", "Active lesions spreading upward", "1.0–1.5 L/ha", "Azole fungicide; alternate with validamycin."),
            _p("Propiconazole 250 EC", "Propiconazole", "Tillering to booting", "0.5–1.0 L/ha", "Also controls other sheath and leaf diseases."),
            _p("Azoxystrobin 250 SC", "Azoxystrobin", "Heavy outbreak on sheaths", "300–400 mL/ha", "Use with reduced nitrogen and wider spacing."),
            _p("Carbendazim 50 WP", "Carbendazim", "Early infection (where label allows)", "500 g–1 kg/ha", "Check compatibility with other pesticides in tank mix."),
        ],
        prevention=["Avoid excessive nitrogen and overcrowding — the two main drivers of sheath blight."],
        farmer_tip="Start management at the first sheath lesions; waiting until the flag leaf is affected is too late.",
    ),
    "sheath rot": _guide(
        key="sheath rot",
        name="Sheath Rot",
        pathogen="Sarocladium oryzae",
        disease_type="fungal",
        local_names=["Panicle discoloration", "Uppama bedi"],
        summary=(
            "Infects the flag-leaf sheath and panicle, causing discoloured or unfilled grains. "
            "Often enters through wounds from insects or field operations."
        ),
        symptoms=[
            "Discolored, rotting flag-leaf sheath near heading",
            "Panicle may not fully emerge or grains appear chalky/discolored",
            "Musty odor in severe cases",
        ],
        procedure=[
            "Scout at booting for sheath discoloration before panicle emergence.",
            "Control stem borers and avoid mechanical injury during operations.",
            "Ensure adequate potassium; avoid unbalanced nitrogen.",
            "Apply carbendazim or propiconazole at booting if infection is confirmed and advised locally.",
            "Use clean seed and destroy infected crop residues.",
        ],
        cultural=[
            "Balanced potassium and proper water depth at heading.",
            "Integrated pest management for stem borers that open entry points.",
        ],
        chemical=["One to two fungicide sprays at booting before panicle emerges."],
        pesticides=[
            _p("Carbendazim 50 WP", "Carbendazim", "Booting stage (before panicle emergence)", "500 g–1 kg/ha", "Common for sheath rot; timing is critical."),
            _p("Propiconazole 250 EC", "Propiconazole", "Booting to early heading", "0.5–1.0 L/ha", "Apply to flag-leaf sheath zone."),
            _p("Mancozeb 80 WP", "Mancozeb", "Early sheath discoloration", "2.0–2.5 kg/ha", "Protectant; start before rot reaches panicle."),
        ],
        prevention=["Healthy panicle emergence depends on protecting the flag leaf from wounds and stress."],
        farmer_tip="Sheath rot and stem borer damage often occur together — check for borer holes on the same plants.",
    ),
    "tungro": _guide(
        key="tungro",
        name="Rice Tungro Disease",
        pathogen="RTBV + RTSV (virus complex); spread by green leafhoppers",
        disease_type="viral",
        local_names=["Tungro", "Dwarf yellow", "Penyakit merah"],
        summary=(
            "Viral disease transmitted by green leafhoppers (Nephotettix spp.). "
            "There is no cure for infected plants — focus on removal and vector control."
        ),
        symptoms=[
            "Stunted plants with yellow or orange leaves, especially on young tillers",
            "Reduced tillering and few or no grains",
            "Leafhopper presence in the field (small green insects that jump when disturbed)",
        ],
        procedure=[
            "Rogue and destroy infected plants immediately — do not compost them near the field.",
            "Control green leafhoppers with approved insecticides only when populations exceed threshold.",
            "Avoid staggered planting that keeps leafhoppers and virus active all season.",
            "Plant tungro-resistant varieties (e.g. NSIC Rc 120, Rc 222 where recommended).",
            "Synchronise planting dates with neighbouring farmers to break the disease cycle.",
            "Remove alternate hosts and weeds along bunds that harbour leafhoppers.",
        ],
        cultural=[
            "Use resistant varieties as the primary strategy — PhilRice lists tungro as a major rainy-season threat.",
            "Shortened crop establishment (e.g. modified dapog) to reduce exposure window where applicable.",
        ],
        chemical=[
            "No pesticide cures tungro — spray only to control green leafhopper vectors (IPM threshold).",
        ],
        pesticides=[
            _p("Imidacloprid 200 SL", "Imidacloprid", "High leafhopper count in first 30 days after transplant", "150–250 mL/ha", "Systemic insecticide; protects young plants from virus spread."),
            _p("Lambda-cyhalothrin 5 EC", "Lambda-cyhalothrin", "Leafhopper surge during vegetative stage", "150–250 mL/ha", "Contact knockdown; avoid spraying during bee activity."),
            _p("Buprofezin 25 SC", "Buprofezin", "Nymph-heavy populations", "300–500 mL/ha", "IGR — slows leafhopper development; good in IPM programs."),
            _p("Thiamethoxam 25 WG", "Thiamethoxam", "Nursery or early field (per label)", "40–80 g/ha", "Seedling/foliar per registration; follow PHI strictly."),
        ],
        prevention=[
            "Never transplant seedlings from tungro-affected nurseries.",
            "Monitor leafhopper numbers weekly during the first 30 days after transplanting.",
        ],
        resistant_varieties=["NSIC Rc 120 and other tungro-resistant varieties for your region."],
        farmer_tip="Removing a few infected hills early saves the whole field — there is no chemical cure for tungro.",
    ),
    "leaf smut": _guide(
        key="leaf smut",
        name="Leaf Smut",
        pathogen="Entyloma oryzae",
        disease_type="fungal",
        local_names=["Leaf smut", "Black spore spots on leaves"],
        summary="Usually a minor leaf disease but can reduce photosynthesis when severe. Spread by spores on seed and wind.",
        symptoms=[
            "Small black, powdery smut sori on leaves (often between veins)",
            "Leaves may show many tiny dark specks",
            "Rarely kills the plant but can weaken growth if severe",
        ],
        procedure=[
            "Use certified disease-free seed for the next planting.",
            "Hot-water seed treatment (52–54 °C for 10 minutes) or approved seed treatment before sowing.",
            "Remove and destroy heavily infected leaves in the nursery and main field.",
            "Avoid excess nitrogen that produces tender, susceptible growth.",
            "Rotate crops in fields with repeated smut history.",
        ],
        cultural=["Field sanitation and seed quality are the main controls PhilRice recommends for smut diseases."],
        chemical=["Field spray rarely needed — seed treatment is the main chemical control."],
        pesticides=[
            _p("Carbendazim (seed treat)", "Carbendazim", "Seed soaking 24 h before sowing", "Per label (g per kg seed)", "Seed treatment, not foliar spray."),
            _p("Thiram 80 WP (seed treat)", "Thiram", "Seed dressing before nursery", "2–3 g per kg seed", "Protects against seed-borne smut spores."),
            _p("Mancozeb 80 WP", "Mancozeb", "Field spray only if leaves heavily speckled", "2.0–2.5 kg/ha", "Secondary option after clean seed fails."),
        ],
        prevention=["Do not save seed from smut-heavy fields."],
        farmer_tip="Leaf smut is often seed-borne — clean seed fixes most problems without field spraying.",
    ),
    "narrow brown leaf spot": _guide(
        key="narrow brown leaf spot",
        name="Narrow Brown Leaf Spot",
        pathogen="Cercospora janseana",
        disease_type="fungal",
        local_names=["Cercospora leaf spot", "NBLS"],
        summary="Leaf spot disease common in nutrient-stressed or water-stressed rice, especially in some upland areas.",
        symptoms=[
            "Narrow, short brown streaks or spots with gray centres, often along veins",
            "Can resemble brown spot but lesions stay narrower and more linear",
            "Worse under low soil fertility or moisture stress",
        ],
        procedure=[
            "Improve soil fertility based on soil test — balanced N, P, K.",
            "Maintain consistent moisture during tillering and panicle initiation.",
            "Apply mancozeb or propiconazole only if lesions are spreading despite fertility correction.",
            "Use tolerant varieties where available.",
        ],
        cultural=["Correct underlying stress — NBLS often indicates the crop is nutritionally or hydrologically weak."],
        chemical=["Spray only after correcting fertility and moisture stress."],
        pesticides=_BROWN_SPOT_FUNGICIDES[:2],
        prevention=["Healthy soil and steady water supply are the long-term solution."],
        farmer_tip="Fix fertility and water first; spraying without correcting stress gives poor results.",
    ),
    "false smut": _guide(
        key="false smut",
        name="False Smut",
        pathogen="Ustilaginoidea virens",
        disease_type="fungal",
        local_names=["Green smut ball", "Kani"],
        summary="Replaces grains with orange to green smut balls at heading. Favoured by high humidity and excess nitrogen.",
        symptoms=[
            "Velvety orange to green balls on grains at heading",
            "Smut balls burst and release spores that infect next seasons",
            "More common with high nitrogen near heading",
        ],
        procedure=[
            "Collect and destroy smut balls before they burst — bag and burn, do not leave in the field.",
            "Reduce nitrogen near heading; split applications earlier in growth.",
            "Apply propiconazole or tebuconazole at booting to early heading if historically a problem.",
            "Never use grain from infected panicles as seed.",
            "Rotate with non-rice crops where false smut is severe.",
        ],
        cultural=["Moderate nitrogen and good field drainage at reproductive stage."],
        chemical=["Azole spray at booting to early heading — before green smut balls appear."],
        pesticides=[
            _p("Propiconazole 250 EC", "Propiconazole", "Booting to early heading", "0.5–1.0 L/ha", "Primary fungicide for false smut prevention."),
            _p("Tebuconazole 250 EW", "Tebuconazole", "Same window as propiconazole", "0.5–1.0 L/ha", "Alternate chemistry if propiconazole was used earlier."),
            _p("Copper hydroxide 77 WP", "Copper hydroxide", "Early heading (organic-compatible option)", "2–3 kg/ha", "Lower toxicity option where effective."),
        ],
        prevention=["Certified seed and destroying smut balls are the most practical farmer actions."],
        farmer_tip="Hand-removing smut balls before harvest protects the next crop more than a late fungicide spray.",
    ),
    "bakanae": _guide(
        key="bakanae",
        name="Bakanae (Foolish Seedling)",
        pathogen="Fusarium fujikuroi",
        disease_type="fungal",
        local_names=["Bakanae", "Foolish seedling", "Lunok"],
        summary="Seed-borne disease causing abnormally tall, pale, weak seedlings. Common when using farm-saved seed.",
        symptoms=[
            "Elongated, thin, pale green seedlings in the nursery",
            "Seedlings may die before or after transplanting",
            "Empty panicles or few grains in surviving plants",
        ],
        procedure=[
            "Reject and destroy abnormal tall seedlings — do not transplant them.",
            "Use certified, fungicide-treated seed from a trusted source.",
            "Hot-water seed treatment (52–54 °C for 10–15 minutes) before sowing where bakanae occurred before.",
            "Carbendazim or thiram seed treatment per label before nursery sowing.",
            "Avoid saving seed from fields with bakanae history.",
        ],
        cultural=["Nursery sanitation and clean water; avoid overcrowded weak seedlings."],
        chemical=["Treat seed before nursery — no effective foliar cure in the main field."],
        pesticides=[
            _p("Carbendazim 50 WP (seed treat)", "Carbendazim", "Seed soak or dust before sowing", "Per label (g per kg seed)", "Most common bakanae seed treatment in the Philippines."),
            _p("Thiram 80 WP (seed treat)", "Thiram", "Seed dressing alternative", "2–3 g per kg seed", "Use certified seed plus treatment for best results."),
            _p("Benomyl / thiophanate-methyl", "Benzimidazole", "Where still FPA-registered for seed", "Per label", "Confirm registration; hot-water soak (52–54 °C, 10–15 min) also helps."),
        ],
        prevention=["Bakanae is prevented at seed, not cured in the main field."],
        farmer_tip="One pale, extra-tall seedling in the tray can mean infected seed lot — replace seed, don't just pull one plant.",
    ),
    "hispa": _guide(
        key="hispa",
        name="Rice Hispa",
        pathogen="Dicladispa armigera (insect pest)",
        disease_type="pest",
        local_names=["Hispa", "Coleoptera leaf scraper"],
        summary="Adults and larvae scrape leaf tissue, leaving white streaks. Not a disease but included in field protection guides.",
        symptoms=[
            "White, papery scraped areas on leaves (feeding marks)",
            "Damaged leaf tips in nursery beds",
            "Adult beetles visible on leaves",
        ],
        procedure=[
            "Clip and destroy damaged leaf tips in the nursery when infestation is light.",
            "Scout weekly; treat only when damage exceeds local economic threshold.",
            "Apply recommended insecticide (e.g. chlorpyrifos, imidacloprid) per label if threshold is reached.",
            "Avoid unnecessary broad-spectrum sprays that kill natural enemies.",
            "Maintain proper spacing for air flow.",
        ],
        cultural=["Early clipping in nursery; conserve spiders and parasitoids."],
        chemical=["Spray only when scouting shows damaging populations — follow IPM thresholds."],
        pesticides=[
            _p("Chlorpyrifos 40 EC", "Chlorpyrifos", "Above economic threshold in nursery/main field", "1.0–1.5 L/ha", "Check local restrictions; avoid runoff to waterways."),
            _p("Imidacloprid 200 SL", "Imidacloprid", "Adult hispa on leaves", "150–250 mL/ha", "Systemic; lower impact on some predators than broad organophosphates."),
            _p("Fipronil 0.3 GR", "Fipronil", "Granular at base (where registered)", "7–10 kg/ha", "Soil/granular application per label."),
            _p("Cartap hydrochloride 50 SP", "Cartap hydrochloride", "Foliar spray at threshold", "1–1.5 kg/ha", "Common rice insecticide; follow PHI."),
        ],
        prevention=["Healthy, unstressed plants tolerate light hispa feeding better."],
        farmer_tip="Light scraping damage often needs no spray — look for beetles and percent of leaves affected first.",
    ),
    "dead heart": _guide(
        key="dead heart",
        name="Dead Heart (Stem Borer)",
        pathogen="Scirpophaga / Chilo spp. (stem borer larvae)",
        disease_type="pest",
        local_names=["Dead heart", "White head (older plants)"],
        summary="Larvae bore into stems; central shoot dies and can be pulled easily. Major insect problem in Philippine rice.",
        symptoms=[
            "Central tiller turns brown and dies (dead heart) at vegetative stage",
            "White empty panicle at heading (white head) from panicle-feeding damage",
            "Small entry holes and frass near the base of tillers",
        ],
        procedure=[
            "Pull out and destroy dead-heart tillers to kill larvae inside.",
            "Install pheromone traps to monitor moth flights and time control.",
            "Apply granular insecticide (cartap, chlorantraniliprole) at tillering per LGU recommendation.",
            "Avoid ratoon crops in heavily infested fields.",
            "Encourage egg parasitoids by reducing unnecessary insecticide sprays.",
        ],
        cultural=[
            "Synchronised planting, field sanitation, and removing stubble that harbours larvae.",
        ],
        chemical=["Apply granular insecticide at tillering when egg masses or dead hearts exceed threshold."],
        pesticides=[
            _p("Cartap hydrochloride 4 GR", "Cartap hydrochloride", "Tillering stage (granular)", "7–10 kg/ha", "Broadcast into standing water; standard stem borer control."),
            _p("Chlorantraniliprole 0.4 GR", "Chlorantraniliprole", "Tillering to early booting", "4–6 kg/ha", "Low-toxicity option; effective on stem borer larvae."),
            _p("Fipronil 0.3 GR", "Fipronil", "Tillering (per label)", "7–10 kg/ha", "Granular at plant base; one application often sufficient."),
            _p("Abamectin 1.8 EC", "Abamectin", "Foliar when moths peak (trap-guided)", "300–500 mL/ha", "Use with pheromone traps to time sprays."),
        ],
        prevention=["Stem borer management starts at tillering — white heads at harvest mean damage weeks earlier."],
        farmer_tip="One dead heart tiller today can mean many white heads at harvest — rogue early.",
    ),
    "physiological stress": _guide(
        key="physiological stress",
        name="Minor Physiological Stress",
        pathogen="No pathogen — abiotic stress (water, nutrition, heat, or salinity)",
        disease_type="physiological",
        local_names=["Dry tip", "Leaf tip burn"],
        summary=(
            "The leaf shows minor discoloration (dry brown tip, slight yellowing, or "
            "water marks) but no spreading lesions of blast, brown spot, or bacterial "
            "blight. This is usually normal senescence or mild abiotic stress — no "
            "pesticide is needed."
        ),
        symptoms=[
            "Dry, brown leaf tip on an otherwise green blade (tip burn / senescence)",
            "Slight, even yellowing without distinct spots, streaks, or powdery growth",
            "Water or mud marks that wipe off and do not spread day to day",
            "No water-soaked margins, spindle-shaped lesions, or oval brown spots",
        ],
        procedure=[
            "Do NOT spray fungicide or bactericide — there is no pathogen to control.",
            "Check water management: avoid alternating drought and deep flooding.",
            "Check nutrition: potassium deficiency and salinity both cause leaf-tip burn; "
            "apply fertiliser per soil test or PhilRice Nutrient Manager.",
            "Re-scout the same plants after 3–5 days: stress marks stay stable, while "
            "true disease lesions enlarge, multiply, or develop halos.",
            "If new spots or streaks appear and spread, re-scan the leaf or consult "
            "your municipal agriculturist (MAO).",
        ],
        cultural=[
            "Stable irrigation and balanced fertilisation prevent most tip burn and yellowing.",
        ],
        prevention=[
            "Mild leaf-tip drying on older leaves is normal as the plant matures — "
            "monitor, but do not treat it as disease.",
        ],
        farmer_tip=(
            "A brown tip on a green leaf is usually thirst or nutrition, not infection — "
            "watch it for a few days before spending on chemicals."
        ),
    ),
    "healthy": _guide(
        key="healthy",
        name="Healthy Rice Plant",
        pathogen="No pathogen detected",
        disease_type="healthy",
        summary="The leaf appears healthy. Continue good agronomic practices and regular scouting.",
        procedure=[
            "Maintain proper water depth and drainage suited to your growth stage.",
            "Apply fertiliser based on soil test and PhilRice Nutrient Manager guidance.",
            "Scout twice weekly during the rainy season for early signs of BLB, blast, and tungro.",
            "Keep field bunds clean and remove weeds that host pests and diseases.",
        ],
        cultural=["Balanced nutrition and timely planting reduce surprise outbreaks later in the season."],
        prevention=["Prevention is cheaper than cure — monitor neighbouring fields during humid weather."],
        farmer_tip="Healthy today does not mean disease-free tomorrow — keep scouting through heading.",
    ),
}


def _normalize_key(disease_name: str) -> str:
    return disease_name.strip().lower()


def get_farmer_guide(disease_name: str) -> FarmerGuide | None:
    """Return structured PhilRice-aligned guide for a disease label, or None."""
    if not disease_name or not disease_name.strip():
        return None
    key = _normalize_key(disease_name)
    if key in GUIDES:
        return GUIDES[key]
    for known_key, guide in GUIDES.items():
        if known_key in key or key in known_key:
            return guide
    return None


_PATHOGEN_LABELS = {
    "fil": {
        "RTBV + RTSV (virus complex); spread by green leafhoppers": (
            "RTBV + RTSV (virus complex); ikinakalat ng green leafhoppers"
        ),
        "Dicladispa armigera (insect pest)": "Dicladispa armigera (peste / insekto)",
        "Scirpophaga / Chilo spp. (stem borer larvae)": (
            "Scirpophaga / Chilo spp. (larva ng stem borer)"
        ),
        "No pathogen — abiotic stress (water, nutrition, heat, or salinity)": (
            "Walang pathogen — abiotic stress (tubig, nutrisyon, init, o asin)"
        ),
        "No pathogen detected": "Walang nakitang pathogen",
    },
    "ilo": {
        "RTBV + RTSV (virus complex); spread by green leafhoppers": (
            "RTBV + RTSV (virus complex); maiwaras dagiti green leafhoppers"
        ),
        "Dicladispa armigera (insect pest)": "Dicladispa armigera (peste / insekto)",
        "Scirpophaga / Chilo spp. (stem borer larvae)": (
            "Scirpophaga / Chilo spp. (larva ti stem borer)"
        ),
        "No pathogen — abiotic stress (water, nutrition, heat, or salinity)": (
            "Awan pathogen — abiotic stress (danum, nutrision, pudot, wenno asin)"
        ),
        "No pathogen detected": "Awan ti nasarakan a pathogen",
    },
}


def get_application_guide(lang: str = "en") -> dict:
    from .i18n import normalize_lang

    code = normalize_lang(lang)
    if code == "ilo":
        return APPLICATION_GUIDE_ILO
    if code == "fil":
        return APPLICATION_GUIDE_FIL
    return APPLICATION_GUIDE


def localize_farmer_guide(guide: FarmerGuide | None, lang: str = "en") -> FarmerGuide | None:
    """Return a language-specific copy so English module data is never mutated."""
    from .i18n import normalize_lang

    if guide is None:
        return guide

    code = normalize_lang(lang)
    if code == "en":
        return guide

    if code == "ilo":
        from .i18n_guides_ilo import GUIDE_TEXT_ILO as guide_text
        from .i18n_pesticides import PESTICIDE_TEXT_ILO as pesticide_text

        apply_map = _APPLY_BY_METHOD_ILO
        method_labels = _METHOD_LABELS_ILO
        pathogen_map = _PATHOGEN_LABELS["ilo"]
    else:
        from .i18n_guides import GUIDE_TEXT_FIL as guide_text
        from .i18n_pesticides import PESTICIDE_TEXT_FIL as pesticide_text

        apply_map = _APPLY_BY_METHOD_FIL
        method_labels = _METHOD_LABELS_FIL
        pathogen_map = _PATHOGEN_LABELS["fil"]

    localized = deepcopy(guide)
    text = guide_text.get(localized.key) or guide_text.get(_normalize_key(localized.key))
    if text:
        for field_name in ("name", "summary", "farmer_tip"):
            if field_name in text:
                setattr(localized, field_name, text[field_name])
        for field_name in (
            "symptoms",
            "procedure",
            "cultural",
            "chemical",
            "prevention",
            "resistant_varieties",
            "local_names",
        ):
            if field_name in text:
                setattr(localized, field_name, list(text[field_name]))

    if localized.pathogen in pathogen_map:
        localized.pathogen = pathogen_map[localized.pathogen]

    timing_map = pesticide_text.get("__timings__", {})
    note_map = pesticide_text.get("__notes__", {})
    for px in localized.pesticides:
        apply = apply_map.get(px.method)
        if apply:
            px.how_to = list(apply["how_to"])
            px.wet_field = apply["wet_field"]
            px.dry_field = apply["dry_field"]
        if px.method in method_labels:
            px.method_label = method_labels[px.method]
        if px.timing in timing_map:
            px.timing = timing_map[px.timing]
        if px.note in note_map:
            px.note = note_map[px.note]
    return localized


def get_procedure_steps(disease_name: str) -> list[str]:
    """Ordered farmer actions for pipeline / API consumers."""
    guide = get_farmer_guide(disease_name)
    return list(guide.procedure) if guide else []
