"""Filipino farmer-guide copy. Keys match detection.disease_guide.GUIDES."""

GUIDE_TEXT_FIL = {
    "bacterial blight": {
        "name": "Bacterial Leaf Blight (BLB)",
        "summary": (
            "Pangunahing bacterial na sakit sa Pilipinas, lalo na sa tag-ulan. "
            "Nagtutuyo ang dahon mula sa dulo pababa at maaaring bumaba nang malaki ang ani sa mahihinang barayti."
        ),
        "symptoms": [
            "Mga water-soaked na guhit sa gilid ng dahon na nagiging dilaw pagkatapos puti/abo",
            "Nalalanta at natutuyo ang dahon mula sa dulo (kresek) sa batang halaman",
            "Puting bacterial ooze mula sa hiwa ng dahon — ilagay sa malinaw na tubig para suriin",
            "Mas malala kapag sobra ang nitrogen at may nakatayong tubig",
        ],
        "procedure": [
            "Suriin linggu-linggo sa tillering at pamumulaklak; markahan ang mga hotspot sa mapa ng bukid.",
            "Alisin at sunugin o ibaon ang mabigat na apektadong halaman at damo sa paligid ng pilapil.",
            "Patuyuin sandali ang bukid pagkatapos ng malakas na ulan; iwasan ang malalim na baha na nagkakalat ng bacteria.",
            "Hatiin ang paglalagay ng nitrogen — iwasan ang mabigat na late nitrogen na nagpapahina sa halaman.",
            "Sa susunod na tanim, gumamit ng BLB-resistant na barayti (hal. NSIC Rc 222, Rc 402) mula sa certified seed.",
            "Araruhin ang dayami pagkatapos anihin; hayaang matuyo ang bukid bago ang susunod na taniman.",
        ],
        "cultural": [
            "Linisin ang bukid: alisin ang volunteer rice, ratoon, at damo na tahanan ng bacteria.",
            "Mababaw na tubig sa seedbed; magandang drainage sa malakas na pagbaha.",
            "Tamang pagitan ng tanim at balanseng pataba (huwag sobrang nitrogen).",
            "Sabay-sabay magtanim sa kapitbahay para paikliin ang panahon ng sakit.",
        ],
        "chemical": [
            "Sinasabi ng PhilRice na madalas hindi matipid o epektibo ang kemikal laban sa BLB.",
            "Mag-spray lamang kung payo ng MAO — una ang lumalaban na barayti at kalinisan ng bukid.",
        ],
        "prevention": [
            "Magtanim ng certified, walang sakit na binhi mula sa lumalaban na barayti para sa inyong lugar.",
            "Iwasang magtrabaho sa bukid habang basa ang dahon (nakakalat ang bacteria).",
            "Huwag ilipat ang may sakit na dayami o punla sa malulusog na bukid.",
        ],
        "resistant_varieties": [
            "Tanungin ang municipal agriculturist para sa BLB-resistant na barayti sa inyong rehiyon "
            "(hal. NSIC Rc 222, Rc 402 / Tubigan 36 kung inirerekomenda)."
        ],
        "farmer_tip": (
            "Mas epektibo ang pag-iwas kaysa spray: binibigyang-diin ng PhilRice ang lumalaban na barayti, "
            "malinis na bukid, at balanseng nitrogen kaysa routine na antibiotic o copper spray."
        ),
    },
    "blast": {
        "name": "Blast sa Palay",
        "summary": (
            "Isa sa pinakamapaminsalang sakit ng palay sa Pilipinas. Kumakalat ang spores sa hangin "
            "at ulan; lumilitaw ang lesions sa dahon, node, at panicle."
        ),
        "symptoms": [
            "Hugis-diamante o spindle na batik na abo sa gitna at kayumanggi sa gilid (leaf blast)",
            "Nabulok na itim na node na maaaring maputol (node/collar blast)",
            "Walang laman o kalahating puno ang butil; maaaring hindi lumabas ang panicle (neck/panicle blast)",
            "Mas malala sa malamig na gabi, hamog, sikip ng canopy, at sobrang nitrogen",
        ],
        "procedure": [
            "Suriin ang dahon sa tillering at muli sa booting — gamutin nang maaga, hindi pagkatapos masira nang malawakan.",
            "Bawasan ang nitrogen; ilagay nang hati-hati (mas ligtas ang ammonium kaysa sobrang nitrate).",
            "Pagandahin ang drainage sa rainfed na lugar para maiwasan ang tagtuyot na sinusundan ng muling pagbaha.",
            "Iwasang magtrabaho habang basa ang halaman; paluwagin ang pagitan kung masyadong sikip ang canopy.",
            "Sunugin o ibaon ang may sakit na dayami pagkatapos anihin; huwag iwanan sa bukid.",
            "Kung aprubado sa lugar, maglagay ng fungicide sa maagang heading at muli kapag lumalabas ang panicle "
            "(hal. tricyclazole, isoprothiolane — sundin ang label at payo ng LGU).",
        ],
        "cultural": [
            "Magtanim ng blast-resistant na barayti (hal. PSB Rc 10 o inirerekomenda sa inyong lugar).",
            "Maglipat-tanim sa tamang pagitan sa halip na sikip na broadcast.",
            "I-time ang pagtatanim para hindi harapin ng pananim ang mabigat na hamog sa pinaka-mahina nitong yugto.",
        ],
        "chemical": [
            "Maglagay ng fungicide kapag kumakalat ang lesions at mahina ang barayti — dalawang spray sa paligid ng heading.",
        ],
        "prevention": [
            "Gumamit ng walang sakit na binhi at lumalaban na barayti bawat season.",
            "Bantayan ang kapitbahay na bukid; malayo ang biyahe ng blast spores sa mahalumigmig na panahon.",
        ],
        "resistant_varieties": [
            "PSB Rc 10 at iba pang NSIC na barayti na may resistensya sa blast para sa inyong ecosystem."
        ],
        "farmer_tip": "Ang neck blast sa heading ang pinakamalaking bawas sa ani — mag-scout bago lumabas ang panicle.",
    },
    "leaf blast": {
        "name": "Leaf Blast",
        "summary": "Blast sa dahon mula punla hanggang tillering. Ang maagang kontrol ay pumipigil sa pagkalat sa node at panicle.",
        "symptoms": [
            "Maliit na hugis-spindle na lesions na abo sa gitna ng dahon",
            "Maaaring magdikit ang lesions at patayin ang tisyu ng dahon",
            "Madalas unang lumitaw sa ibabang dahon sa mahalumigmig at madilim na bahagi ng bukid",
        ],
        "procedure": [
            "Alisin ang malubhang apektadong dahon o tiller habang lokal pa ang impeksiyon.",
            "Bawasan ang nitrogen; hatiin ang aplikasyon sa halip na isang mabigat na dosis.",
            "Panatilihing matatag ang irigasyon — iwasan ang mahabang pagkabasa ng dahon sa gabi.",
            "Magplano ng fungicide lamang kung kumakalat ang lesions at mahina ang barayti.",
            "Lumipat sa blast-resistant na barayti sa susunod na taniman.",
        ],
        "cultural": [
            "Tamang pagitan at pagkontrol ng damo para sa mas magandang daloy ng hangin.",
            "Iwasan ang irigasyon sa gabi na nagpapanatiling basa ng dahon buong gabi.",
        ],
        "chemical": ["Mag-spray sa unang kumakalat na lesions; ulitin ayon sa label kung nahugasan ng ulan."],
        "prevention": ["Ang lumalaban na barayti at balanseng pataba ang pundasyon ng kontrol sa blast."],
        "farmer_tip": "Ang leaf blast ngayon ay maaaring maging neck blast sa heading — kumilos habang limitado pa ang lesions.",
    },
    "neck blast": {
        "name": "Neck / Panicle Blast",
        "summary": (
            "Ang impeksiyon sa leeg ng panicle ay nagdudulot ng walang lamang butil at tuwirang bawas sa ani. "
            "Kahit 10% na neck infection ay nakakabawas na sa ani at kalidad ng butil."
        ),
        "symptoms": [
            "Kayumanggi hanggang abong lesion na bumabalot sa leeg ng panicle",
            "Maaaring maputol ang panicle o manatili sa loob ng boot",
            "Kalahating puno o chalky na butil",
        ],
        "procedure": [
            "Maglagay ng inirerekomendang fungicide sa panicle initiation at maagang heading kung mahina ang barayti.",
            "Huwag maglagay ng mabigat na nitrogen malapit sa heading.",
            "Markahan ang apektadong bahagi para palitan ng lumalaban na barayti sa susunod na season.",
            "Anihin agad; ibaon o sunugin ang may sakit na dayami.",
        ],
        "cultural": [
            "Magtanim nang sapat ang aga para maiwasan ang rurok ng sakit sa heading.",
            "Gumamit ng lumalaban na barayti sa blast-prone na irrigated at upland na lugar.",
        ],
        "chemical": ["Dalawang spray ng fungicide sa booting at heading ang mahalaga para sa neck/panicle blast."],
        "prevention": ["Nagsisimula ang pamamahala ng neck blast sa pagpili ng barayti at nitrogen ilang linggo bago."],
        "farmer_tip": "Sa yugtong ito ang pinakamalaking bawas sa ani — unahin ang lumalaban na barayti sa blast hotspot.",
    },
    "brown spot": {
        "name": "Brown Spot",
        "summary": (
            "Karaniwang fungal na sakit sa dahon, madalas kaugnay ng mahinang fertility ng lupa, tagtuyot, o mahinang halaman. "
            "Kabilang ito sa mga pangunahing sakit sa dahon ayon sa PhilRice."
        ),
        "symptoms": [
            "Maliit na bilog o oval na kayumangging batik na abo o tan ang gitna sa dahon",
            "Maaaring lumaki at magdikit ang batik; maaaring matuyo ang malubhang apektadong dahon",
            "Nakikita rin sa leaf sheath at butil sa malalang outbreak",
            "Mas karaniwan kapag kulang ang nutrisyon o may moisture stress ang halaman",
        ],
        "procedure": [
            "Kumpirmahin kung brown spot (bilog na lesions na abo ang gitna) at hindi blast (hugis-diamante).",
            "Pagandahin ang fertility ng lupa — maglagay ng balanseng NPK at organic matter ayon sa soil test.",
            "Itama ang kakulangan sa silicon at potassium kung mababa ito sa soil test.",
            "Panatilihin ang mababaw at pantay na irigasyon; iwasan ang matagal na tagtuyot sa tillering at pamumulaklak.",
            "Alisin ang mabigat na apektadong dayami pagkatapos anihin; araruhin ang stubble.",
            "Kung mabilis kumalat, maglagay ng mancozeb o propiconazole ayon sa label at gabay ng LGU.",
            "Sa susunod na season gumamit ng certified, fungicide-treated na binhi mula sa lumalaban o tolerant na barayti.",
        ],
        "cultural": [
            "Balanseng pataba — iwasang gutumin ang pananim tapos lagyan ng sobrang nitrogen sa huli.",
            "Mabuting pamamahala ng tubig; bawasan ang stress sa mahahalagang yugto ng paglaki.",
            "Pagpapalitan ng pananim at kalinisan ng bukid kung bumabalik ang brown spot bawat season.",
        ],
        "chemical": [
            "Mag-spray ng fungicide kapag kumakalat ang batik pagkatapos itama ang fertility at water stress.",
        ],
        "prevention": [
            "Ang malusog at fertil na lupa ang pinakamahusay na depensa — madalas senyales ng nutrient o moisture stress ang brown spot.",
            "Gumamit ng magandang binhi at iwasang magtanim mula sa butil ng mabigat na apektadong bukid.",
        ],
        "resistant_varieties": [
            "Tanungin ang PhilRice o MAO para sa barayti na may magandang tolerance sa brown spot sa inyong lugar."
        ],
        "farmer_tip": (
            "Binibigyang-diin ng PhilRice ang pagtama ng fertility at water stress — hindi aayusin ng fungicide mag-isa "
            "ang mahinang lupa."
        ),
    },
    "sheath blight": {
        "name": "Sheath Blight",
        "summary": (
            "Fungal na sakit na pinalalakas ng mataas na humidity, sikip ng canopy, at sobrang nitrogen. "
            "Umakyat ang lesions mula sa leaf sheath at maaaring masira ang panicle."
        ),
        "symptoms": [
            "Oval o irregular na greenish-gray na lesions na kayumanggi ang gilid sa leaf sheath",
            "Maaaring magdikit ang lesions at mabulok ang sheath; maaaring mamatay ang itaas na dahon",
            "May sclerotia (maliit na maitim na pellet) sa apektadong tisyu",
            "Mas malala sa sikip at sobrang pataba na bukid",
        ],
        "procedure": [
            "Mag-scout sa gilid ng bukid at mababang bahagi kung saan mataas ang humidity.",
            "Bawasan ang densidad sa susunod — mas maluwag na pagitan ay nagpapaganda ng daloy ng hangin.",
            "Bawasan ang nitrogen; iwasan ang malambot at mahinang tangkay.",
            "Patuyuin paminsan-minsan kung masyadong matagal ang nakatayong tubig.",
            "Maglagay ng validamycin, hexaconazole, o propiconazole sa unang lesions sa sheath.",
            "Huwag gamitin ang may sakit na dayami bilang mulch.",
        ],
        "cultural": [
            "Balanseng pataba at inirerekomendang densidad ng tanim.",
            "Paminsan-minsang pagpapatuyo sa irrigated na bukid sa tag-ulan.",
        ],
        "chemical": ["Mag-spray sa unang lesions sa ibabang sheath — hindi pagkatapos maapektuhan ang flag leaf."],
        "prevention": ["Iwasan ang sobrang nitrogen at sikip — ito ang dalawang pangunahing sanhi ng sheath blight."],
        "farmer_tip": "Simulan ang pamamahala sa unang lesions sa sheath; huli na kapag apektado na ang flag leaf.",
    },
    "sheath rot": {
        "name": "Sheath Rot",
        "summary": (
            "Iniimpeksyon ang flag-leaf sheath at panicle, kaya nagkakaroon ng discolored o walang lamang butil. "
            "Madalas pumapasok sa sugat mula sa insekto o gawain sa bukid."
        ),
        "symptoms": [
            "Nababago ang kulay at nabubulok ang flag-leaf sheath malapit sa heading",
            "Maaaring hindi ganap na lumabas ang panicle o chalky/discolored ang butil",
            "May amoy na musty sa malalang kaso",
        ],
        "procedure": [
            "Mag-scout sa booting para sa pagbabago ng kulay ng sheath bago lumabas ang panicle.",
            "Kontrolin ang stem borer at iwasan ang mekanikal na sugat habang nagtatrabaho.",
            "Siyakin ang sapat na potassium; iwasan ang hindi balanseng nitrogen.",
            "Maglagay ng carbendazim o propiconazole sa booting kung kumpirmado at payo ng lokal.",
            "Gumamit ng malinis na binhi at sirain ang apektadong dayami.",
        ],
        "cultural": [
            "Balanseng potassium at tamang lalim ng tubig sa heading.",
            "Integrated pest management para sa stem borer na nagbubukas ng daan ng impeksiyon.",
        ],
        "chemical": ["Isa hanggang dalawang spray ng fungicide sa booting bago lumabas ang panicle."],
        "prevention": ["Nakasalalay ang malusog na paglabas ng panicle sa pagprotekta sa flag leaf mula sa sugat at stress."],
        "farmer_tip": "Madalas magkasama ang sheath rot at stem borer — tingnan ang butas ng borer sa parehong halaman.",
    },
    "tungro": {
        "name": "Sakit na Tungro",
        "summary": (
            "Viral na sakit na dinadala ng green leafhopper (Nephotettix spp.). "
            "Walang lunas para sa apektadong halaman — unahin ang pag-alis at kontrol ng insekto."
        ),
        "symptoms": [
            "Pandak na halaman na dilaw o kahel ang dahon, lalo na sa batang tiller",
            "Kakaunting tiller at kakaunti o walang butil",
            "May leafhopper sa bukid (maliit na berdeng insekto na tumatalon kapag ginulo)",
        ],
        "procedure": [
            "Alisin at sirain agad ang apektadong halaman — huwag i-compost malapit sa bukid.",
            "Kontrolin ang green leafhopper ng aprubadong insecticide lamang kapag lumampas sa threshold.",
            "Iwasan ang staggered na pagtatanim na nagpapanatili ng leafhopper at virus buong season.",
            "Magtanim ng tungro-resistant na barayti (hal. NSIC Rc 120, Rc 222 kung inirerekomenda).",
            "I-synchronize ang petsa ng pagtatanim sa mga kapitbahay para masira ang siklo ng sakit.",
            "Alisin ang damo sa pilapil na tahanan ng leafhopper.",
        ],
        "cultural": [
            "Gumamit ng lumalaban na barayti bilang pangunahing estratehiya — itinuturing ng PhilRice ang tungro na banta sa tag-ulan.",
            "Paikliin ang crop establishment (hal. modified dapog) para bawasan ang panahon ng exposure kung naaangkop.",
        ],
        "chemical": [
            "Walang pestisidyo ang gumagamot sa tungro — mag-spray lamang para kontrolin ang green leafhopper (IPM threshold).",
        ],
        "prevention": [
            "Huwag maglipat-tanim ng punla mula sa nursery na may tungro.",
            "Bantayan ang bilang ng leafhopper linggu-linggo sa unang 30 araw pagkatapos maglipat-tanim.",
        ],
        "resistant_varieties": ["NSIC Rc 120 at iba pang tungro-resistant na barayti para sa inyong rehiyon."],
        "farmer_tip": "Ang maagang pag-alis ng iilang apektadong hill ay nakakapagligtas ng buong bukid — walang kemikal na lunas sa tungro.",
    },
    "leaf smut": {
        "name": "Leaf Smut",
        "summary": "Karaniwang menor na sakit sa dahon ngunit maaaring bawasan ang photosynthesis kung malala. Kumakalat sa spores sa binhi at hangin.",
        "symptoms": [
            "Maliit na itim, powdery na smut sori sa dahon (madalas sa pagitan ng ugat)",
            "Maraming maliliit na maitim na tuldok sa dahon",
            "Bihirang pumatay sa halaman ngunit maaaring humina ang paglaki kung malala",
        ],
        "procedure": [
            "Gumamit ng certified na walang sakit na binhi sa susunod na taniman.",
            "Hot-water seed treatment (52–54 °C sa loob ng 10 minuto) o aprubadong seed treatment bago maghasik.",
            "Alisin at sirain ang mabigat na apektadong dahon sa nursery at pangunahing bukid.",
            "Iwasan ang sobrang nitrogen na nagdudulot ng malambot at mahinang paglaki.",
            "Magpalit ng pananim sa bukid na paulit-ulit ang smut.",
        ],
        "cultural": ["Ang kalinisan ng bukid at kalidad ng binhi ang pangunahing kontrol na inirerekomenda ng PhilRice para sa smut."],
        "chemical": ["Bihirang kailanganin ang spray sa bukid — ang seed treatment ang pangunahing kemikal na kontrol."],
        "prevention": ["Huwag magtanim ng binhi mula sa bukid na maraming smut."],
        "farmer_tip": "Madalas seed-borne ang leaf smut — malinis na binhi ang solusyon sa karamihan, hindi spray sa bukid.",
    },
    "narrow brown leaf spot": {
        "name": "Narrow Brown Leaf Spot",
        "summary": "Sakit sa dahon na karaniwan sa palay na kulang sa nutrisyon o tubig, lalo na sa ilang upland na lugar.",
        "symptoms": [
            "Makitid, maikling kayumangging guhit o batik na abo ang gitna, madalas sa ugat",
            "Maaaring kamukha ng brown spot ngunit mas makitid at mas tuwid ang lesions",
            "Mas malala sa mababang fertility o moisture stress",
        ],
        "procedure": [
            "Pagandahin ang fertility ng lupa ayon sa soil test — balanseng N, P, K.",
            "Panatilihin ang tuloy-tuloy na moisture sa tillering at panicle initiation.",
            "Maglagay ng mancozeb o propiconazole lamang kung kumakalat pa rin ang lesions pagkatapos itama ang fertility.",
            "Gumamit ng tolerant na barayti kung available.",
        ],
        "cultural": ["Itama ang pinagmulang stress — madalas senyales ng mahinang nutrisyon o tubig ang NBLS."],
        "chemical": ["Mag-spray lamang pagkatapos itama ang fertility at moisture stress."],
        "prevention": ["Ang malusog na lupa at tuloy-tuloy na tubig ang pangmatagalang solusyon."],
        "farmer_tip": "Ayusin muna ang fertility at tubig; mahina ang resulta ng spray kung hindi naitama ang stress.",
    },
    "false smut": {
        "name": "False Smut",
        "summary": "Pinapalitan ang butil ng orange hanggang berdeng smut ball sa heading. Pinalalakas ng mataas na humidity at sobrang nitrogen.",
        "symptoms": [
            "Malambot na orange hanggang berdeng bola sa butil sa heading",
            "Pumutok ang smut ball at naglalabas ng spores na nakakahawa sa susunod na season",
            "Mas karaniwan kapag mataas ang nitrogen malapit sa heading",
        ],
        "procedure": [
            "Kolektahin at sirain ang smut ball bago pumutok — i-bag at sunugin, huwag iwanan sa bukid.",
            "Bawasan ang nitrogen malapit sa heading; hatiin ang aplikasyon nang mas maaga.",
            "Maglagay ng propiconazole o tebuconazole sa booting hanggang maagang heading kung dati nang problema.",
            "Huwag gamitin bilang binhi ang butil mula sa apektadong panicle.",
            "Magpalit ng hindi palay na pananim kung malala ang false smut.",
        ],
        "cultural": ["Katamtamang nitrogen at magandang drainage sa reproductive stage."],
        "chemical": ["Azole spray sa booting hanggang maagang heading — bago lumitaw ang berdeng smut ball."],
        "prevention": ["Ang certified seed at pagwasak ng smut ball ang pinakapraktikal na gawain ng magsasaka."],
        "farmer_tip": "Ang pag-alis ng smut ball sa kamay bago anihin ay mas nakakatulong sa susunod na tanim kaysa huling fungicide spray.",
    },
    "bakanae": {
        "name": "Bakanae (Foolish Seedling)",
        "summary": "Sakit na dala ng binhi na nagdudulot ng sobrang taas, maputla, at mahinang punla. Karaniwan kapag sariling binhi ang gamit.",
        "symptoms": [
            "Mahaba, manipis, maputlang berdeng punla sa nursery",
            "Maaaring mamatay ang punla bago o pagkatapos maglipat-tanim",
            "Walang laman o kakaunting butil sa natitirang halaman",
        ],
        "procedure": [
            "Tanggihan at sirain ang abnormal na matataas na punla — huwag itanim.",
            "Gumamit ng certified, fungicide-treated na binhi mula sa mapagkakatiwalaang pinagmulan.",
            "Hot-water seed treatment (52–54 °C sa loob ng 10–15 minuto) bago maghasik kung may bakanae dati.",
            "Carbendazim o thiram seed treatment ayon sa label bago maghasik sa nursery.",
            "Iwasang magtanim ng binhi mula sa bukid na may kasaysayan ng bakanae.",
        ],
        "cultural": ["Kalinisan ng nursery at malinis na tubig; iwasan ang sikip at mahinang punla."],
        "chemical": ["Gamutin ang binhi bago ang nursery — walang epektibong foliar na lunas sa pangunahing bukid."],
        "prevention": ["Napipigilan ang bakanae sa binhi, hindi ginagamot sa pangunahing bukid."],
        "farmer_tip": "Isang maputla at sobrang taas na punla sa tray ay maaaring senyales ng infected na lot ng binhi — palitan ang binhi, huwag lang bunutin ang isa.",
    },
    "hispa": {
        "name": "Rice Hispa",
        "summary": "Kumakamot ang adult at larva sa dahon, nag-iiwan ng puting guhit. Hindi sakit ngunit kabilang sa gabay sa proteksyon ng bukid.",
        "symptoms": [
            "Puti, papery na kinalmot na bahagi sa dahon (feeding marks)",
            "Nasirang dulo ng dahon sa nursery",
            "Nakikitang adult beetle sa dahon",
        ],
        "procedure": [
            "Gupitin at sirain ang nasirang dulo ng dahon sa nursery kapag magaan pa ang infestation.",
            "Mag-scout linggu-linggo; gamutin lamang kung lumampas sa lokal na economic threshold.",
            "Maglagay ng inirerekomendang insecticide (hal. chlorpyrifos, imidacloprid) ayon sa label kung naabot ang threshold.",
            "Iwasan ang hindi kinakailangang broad-spectrum spray na pumapatay sa natural enemies.",
            "Panatilihin ang tamang pagitan para sa daloy ng hangin.",
        ],
        "cultural": ["Maagang paggupit sa nursery; pangalagaan ang gagamba at parasitoid."],
        "chemical": ["Mag-spray lamang kapag nagpapakita ang scouting ng mapaminsalang populasyon — sundin ang IPM threshold."],
        "prevention": ["Mas nagtitiis ang malusog, walang stress na halaman sa magaan na pagkain ng hispa."],
        "farmer_tip": "Madalas hindi kailangan ng spray ang magaan na kagat — tingnan muna ang beetle at porsyento ng apektadong dahon.",
    },
    "dead heart": {
        "name": "Dead Heart (Stem Borer)",
        "summary": "Bumubutas ang larva sa tangkay; namamatay ang gitnang shoot at madaling mabunot. Pangunahing problema sa insekto sa palay sa Pilipinas.",
        "symptoms": [
            "Nagmumula at namamatay ang gitnang tiller (dead heart) sa vegetative stage",
            "Puting walang lamang panicle sa heading (white head) mula sa panicle-feeding damage",
            "Maliit na butas at frass malapit sa base ng tiller",
        ],
        "procedure": [
            "Bunutin at sirain ang dead-heart tiller para patayin ang larva sa loob.",
            "Maglagay ng pheromone trap para bantayan ang moth at i-time ang kontrol.",
            "Maglagay ng granular insecticide (cartap, chlorantraniliprole) sa tillering ayon sa rekomendasyon ng LGU.",
            "Iwasan ang ratoon crop sa mabigat na infested na bukid.",
            "Hikayatin ang egg parasitoid sa pamamagitan ng pagbabawas ng hindi kinakailangang insecticide spray.",
        ],
        "cultural": [
            "Sabay-sabay na pagtatanim, kalinisan ng bukid, at pag-alis ng stubble na tahanan ng larva.",
        ],
        "chemical": ["Maglagay ng granular insecticide sa tillering kapag lumampas sa threshold ang egg mass o dead heart."],
        "prevention": ["Nagsisimula ang pamamahala ng stem borer sa tillering — ang white head sa ani ay ibig sabihin nasira na linggo ang nakalipas."],
        "farmer_tip": "Isang dead heart tiller ngayon ay maaaring maraming white head sa ani — alisin nang maaga.",
    },
    "physiological stress": {
        "name": "Menor na Physiological Stress",
        "summary": (
            "May menor na pagbabago ng kulay ang dahon (tuyo at kayumangging dulo, bahagyang pagdilaw, o "
            "bakas ng tubig) ngunit walang kumakalat na lesions ng blast, brown spot, o bacterial blight. "
            "Karaniwan itong normal na senescence o magaan na abiotic stress — hindi kailangan ng pestisidyo."
        ),
        "symptoms": [
            "Tuyo at kayumangging dulo ng dahon sa berdeng talim (tip burn / senescence)",
            "Bahagyang pantay na pagdilaw na walang malinaw na batik, guhit, o powdery growth",
            "Bakas ng tubig o putik na nabubura at hindi kumakalat araw-araw",
            "Walang water-soaked na gilid, hugis-spindle na lesions, o oval na kayumangging batik",
        ],
        "procedure": [
            "HUWAG mag-spray ng fungicide o bactericide — walang pathogen na kontrolin.",
            "Tingnan ang pamamahala ng tubig: iwasan ang palit-palit na tagtuyot at malalim na baha.",
            "Tingnan ang nutrisyon: ang kakulangan sa potassium at salinity ay nagdudulot ng leaf-tip burn; "
            "maglagay ng pataba ayon sa soil test o PhilRice Nutrient Manager.",
            "Suriin ulit ang parehong halaman pagkatapos ng 3–5 araw: nananatili ang bakas ng stress, "
            "habang lumalaki, dumadami, o nagkakaroon ng halo ang totoong sakit.",
            "Kung lumitaw at kumalat ang bagong batik o guhit, i-scan ulit ang dahon o kumonsulta "
            "sa municipal agriculturist (MAO).",
        ],
        "cultural": [
            "Matatag na irigasyon at balanseng pataba ang pumipigil sa karamihan ng tip burn at pagdilaw.",
        ],
        "prevention": [
            "Normal ang magaan na pagkatuyo ng dulo ng lumang dahon habang tumatanda ang halaman — "
            "bantayan, ngunit huwag ituring na sakit.",
        ],
        "farmer_tip": (
            "Ang kayumangging dulo sa berdeng dahon ay madalas uhaw o nutrisyon, hindi impeksiyon — "
            "bantayan ng ilang araw bago gumastos sa kemikal."
        ),
    },
    "healthy": {
        "name": "Malusog na Palay",
        "summary": "Mukhang malusog ang dahon. Ipagpatuloy ang mabuting gawain sa bukid at regular na pag-scout.",
        "procedure": [
            "Panatilihin ang tamang lalim ng tubig at drainage ayon sa yugto ng paglaki.",
            "Maglagay ng pataba ayon sa soil test at gabay ng PhilRice Nutrient Manager.",
            "Mag-scout dalawang beses sa isang linggo sa tag-ulan para sa maagang senyales ng BLB, blast, at tungro.",
            "Panatilihing malinis ang pilapil at alisin ang damo na tahanan ng peste at sakit.",
        ],
        "cultural": ["Ang balanseng nutrisyon at tamang oras ng pagtatanim ay nagbabawas ng sorpresang outbreak sa huling bahagi ng season."],
        "prevention": ["Mas mura ang pag-iwas kaysa gamot — bantayan ang kapitbahay na bukid sa mahalumigmig na panahon."],
        "farmer_tip": "Ang malusog ngayon ay hindi nangangahulugang walang sakit bukas — magpatuloy sa pag-scout hanggang heading.",
    },
}
