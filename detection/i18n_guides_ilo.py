"""Ilocano farmer-guide copy. Keys match detection.disease_guide.GUIDES."""

GUIDE_TEXT_ILO = {
    "bacterial blight": {
        "name": "Bacterial Leaf Blight (BLB)",
        "summary": (
            "Kangrunaan a sakit a gapuanan ti bacteria ditoy Pilipinas, nangnangruna iti panawen ti tudo. "
            "Magango ti bulong manipud iti murdong nga agpababa ket mabalin nga agkissay iti dakkel ti apit "
            "kadagiti nakapuy a barayti."
        ),
        "symptoms": [
            "Water-soaked a guhit iti igid ti bulong nga agbalin a duyaw sa agbalin a puraw wenno kolor dapo",
            "Aglaylay ken magango ti bulong manipud iti murdong (kresek) kadagiti ubbing a mula",
            "Puraw a bacterial ooze nga agtedted manipud iti napukis a bulong — ikabil iti nalitnaw a danum tapno masukimat",
            "Ad-adda a dumakkel no nalabes ti nitrogen ken adda naurnong a danum",
        ],
        "procedure": [
            "Agsukimatkayo kada lawas bayat ti tillering ken panagsabong; markaan dagiti hotspot iti mapa ti talon.",
            "Ikkaten ken puoran wenno itabon dagiti nakaro nga apektado a mula ken dagiti ruot iti aglawlaw ti tambak.",
            "Pagpaawanen ti danum iti apagbiit kalpasan ti napigsa a tudo; liklikan ti nauneg a layus a mangiwaras iti bacteria.",
            "Bingayen ti panangikabil iti nitrogen — liklikan ti nabuslon ken naladaw a nitrogen a mangpakapuy iti mula.",
            "Iti sumaruno a panagmula, agusar iti barayti a resistant iti BLB (kas iti NSIC Rc 222, Rc 402) manipud iti certified seed.",
            "Aradoen ti garami kalpasan ti panagapit; palubosan a magango ti talon sakbay ti sumaruno a panagmula.",
        ],
        "cultural": [
            "Dalusan ti talon: ikkaten dagiti volunteer rice, ratoon, ken ruot a paggianan ti bacteria.",
            "Nababaw a danum iti pagbunubonan; nasayaat a panagayus ti danum no napigsa ti layus.",
            "Umno a nagbaetan ti mula ken balanse nga abono (saan a nalabes ti nitrogen).",
            "Agsasabay a panagmula kadagiti kaarruba tapno umababa ti panawen ti sakit.",
        ],
        "chemical": [
            "Kuna ti PhilRice a masansan a saan a nalaka wenno saan nga epektibo ti kemikal maibusor iti BLB.",
            "Agispray laeng no ibalakad ti MAO — ipangruna ti resistant a barayti ken ti kinadalus ti talon.",
        ],
        "prevention": [
            "Agmula iti certified ken awan sakitna a bin-i manipud kadagiti resistant a barayti para iti lugaryo.",
            "Liklikan ti agtrabaho iti talon bayat a nabasa ti bulong (agwaras ti bacteria).",
            "Saanyo nga iyakar ti naspol a garami wenno bunubon kadagiti nasalun-at a talon.",
        ],
        "resistant_varieties": [
            "Damagenyo iti municipal agriculturist ti barayti a resistant iti BLB para iti rehionyo "
            "(kas iti NSIC Rc 222, Rc 402 / Tubigan 36 no maidasig)."
        ],
        "farmer_tip": (
            "Ad-adda nga epektibo ti panangliklik ngem ti panagispray: ipangpangruna ti PhilRice ti resistant a barayti, "
            "nadalus a talon, ken balanse a nitrogen imbes a ti gagangay nga antibiotic wenno copper spray."
        ),
    },
    "blast": {
        "name": "Blast iti Pagay",
        "summary": (
            "Maysa kadagiti kadaksan a sakit ti pagay ditoy Pilipinas. Agwaras dagiti spores babaen ti angin "
            "ken tudo; agparang dagiti lesion iti bulong, node, ken panicle."
        ),
        "symptoms": [
            "Kasla diamante wenno spindle a marka a kolor dapo ti tengngana ken kolor kape ti igidna (leaf blast)",
            "Naruket ken nangisit a node a mabalin a mapugsat (node/collar blast)",
            "Awan lamanna wenno kagudua laeng ti laon ti bukel; mabalin a saan a rummuar ti panicle (neck/panicle blast)",
            "Ad-adda a nakaro no nalam-ek ti rabii, adu ti linnaaw, nairut ti canopy, ken nalabes ti nitrogen",
        ],
        "procedure": [
            "Kitaen ti bulong iti tillering ken manen iti booting — agasan a nasapa, saan a kalpasan ti nalawa a dadael.",
            "Kissayan ti nitrogen; ikabil a biningay (nasaysayaat ti ammonium ngem ti nalabes a nitrate).",
            "Pasayaaten ti panagayus ti danum kadagiti rainfed a lugar tapno maliklikan ti kalgaw a sarunuen ti layus manen.",
            "Liklikan ti agtrabaho bayat a nabasa ti mula; palawaen ti nagbaetan no nairut unay ti canopy.",
            "Puoran wenno itabon ti naspol a garami kalpasan ti panagapit; saan nga ibati iti talon.",
            "No naanamongan iti lugaryo, mangikabil iti fungicide iti nasapa a heading ken manen no rumrummuar ti panicle "
            "(kas iti tricyclazole, isoprothiolane — surotenyo ti label ken ti balakad ti LGU).",
        ],
        "cultural": [
            "Agmula iti barayti a resistant iti blast (kas iti PSB Rc 10 wenno ti maidasig iti lugaryo).",
            "Agraep iti umno a nagbaetan imbes a nairut a broadcast.",
            "Ituon ti panagmula tapno saan a masabat ti mula ti napigsa a linnaaw iti kanakapuyan a yugtona.",
        ],
        "chemical": [
            "Mangikabil iti fungicide no agwarwaras dagiti lesion ken nakapuy ti barayti — dua nga ispray iti aglawlaw ti heading.",
        ],
        "prevention": [
            "Agusar iti bin-i nga awan sakitna ken iti resistant a barayti iti tunggal panawen.",
            "Bantayan dagiti talon dagiti kaarruba; adayo ti pagturongan dagiti blast spores no nabasa ti tiempo.",
        ],
        "resistant_varieties": [
            "PSB Rc 10 ken dadduma pay a barayti ti NSIC nga addaan resistensia iti blast para iti ecosystem-yo."
        ],
        "farmer_tip": "Ti neck blast iti heading ti kadakkelan a mangkissay iti apit — agsukimat sakbay a rummuar ti panicle.",
    },
    "leaf blast": {
        "name": "Leaf Blast",
        "summary": "Blast iti bulong manipud bunubon agingga iti tillering. Ti nasapa a panangkontrol ket manglapped iti panagwaras iti node ken panicle.",
        "symptoms": [
            "Babassit a kasla spindle a lesion a kolor dapo ti tengnga ti bulong",
            "Mabalin nga agdikket dagiti lesion ket matay ti tisiu ti bulong",
            "Masansan nga umuna nga agparang kadagiti akin-baba a bulong iti nabasa ken nasipnget a paset ti talon",
        ],
        "procedure": [
            "Ikkaten dagiti nakaro nga apektado a bulong wenno tiller bayat a lokal pay laeng ti impeksion.",
            "Kissayan ti nitrogen; bingayen ti aplikasion imbes a maymaysa a nabuslon a dosis.",
            "Pagtalinaeden a natibker ti irigasion — liklikan ti nabayag a pannakabasa ti bulong iti rabii.",
            "Agplano iti fungicide no agwarwaras laeng dagiti lesion ken nakapuy ti barayti.",
            "Agsukat iti barayti a resistant iti blast iti sumaruno a panagmula.",
        ],
        "cultural": [
            "Umno a nagbaetan ken panangkontrol iti ruot para iti nasaysayaat a panagayus ti angin.",
            "Liklikan ti panagirigar iti rabii a mangpabasa iti bulong iti agpatnag.",
        ],
        "chemical": ["Agispray iti damo nga agwaras dagiti lesion; uliten sigun iti label no nabugguan ti tudo."],
        "prevention": ["Ti resistant a barayti ken ti balanse nga abono ti pundasion ti panangkontrol iti blast."],
        "farmer_tip": "Ti leaf blast ita ket mabalin nga agbalin a neck blast inton heading — agtignay bayat a bassit pay dagiti lesion.",
    },
    "neck blast": {
        "name": "Neck / Panicle Blast",
        "summary": (
            "Ti impeksion iti tengnged ti panicle ket pakaigapuan ti awan lamanna a bukel ken direkta a pannakakissay ti apit. "
            "Uray ti 10% a neck infection ket makakissay iti apit ken iti kalidad ti bukel."
        ),
        "symptoms": [
            "Kolor kape agingga kolor dapo a lesion a mangbalkot iti tengnged ti panicle",
            "Mabalin a mapugsat ti panicle wenno agtalinaed iti uneg ti boot",
            "Kagudua laeng ti laon wenno chalky a bukel",
        ],
        "procedure": [
            "Mangikabil iti maidasig a fungicide iti panicle initiation ken iti nasapa a heading no nakapuy ti barayti.",
            "Saan a mangikabil iti nabuslon a nitrogen no asidegen ti heading.",
            "Markaan dagiti apektado a paset tapno masukatan iti resistant a barayti iti sumaruno a panawen.",
            "Agapit a dagus; itabon wenno puoran ti naspol a garami.",
        ],
        "cultural": [
            "Agmula a nasapa tapno maliklikan ti kangatuan a panawen ti sakit iti heading.",
            "Agusar iti resistant a barayti kadagiti irrigated ken upland a lugar a masansan ti blast.",
        ],
        "chemical": ["Dua nga ispray ti fungicide iti booting ken heading ti napateg maibusor iti neck/panicle blast."],
        "prevention": ["Mangrugi ti panangtaming iti neck blast iti panagpili iti barayti ken iti nitrogen sumagmamano a lawas sakbayna."],
        "farmer_tip": "Iti daytoy a yugto ti kadakkelan a pannakakissay ti apit — ipangruna ti resistant a barayti kadagiti blast hotspot.",
    },
    "brown spot": {
        "name": "Brown Spot",
        "summary": (
            "Gagangay a sakit ti bulong a gapuanan ti fungus, masansan a nainaig iti nakapuy a kinabaknang ti daga, "
            "kalgaw, wenno nakapuy a mula. Maysa daytoy kadagiti kangrunaan a sakit ti bulong sigun iti PhilRice."
        ),
        "symptoms": [
            "Babassit a nagtimbukel wenno oval a marka a kolor kape nga addaan kolor dapo wenno tan a tengnga iti bulong",
            "Mabalin a dumakkel ken agdikket dagiti marka; mabalin a magango dagiti nakaro nga apektado a bulong",
            "Makita met iti leaf sheath ken iti bukel no nakaro ti outbreak",
            "Ad-adda a mapasamak no kurang ti sustansia wenno adda moisture stress ti mula",
        ],
        "procedure": [
            "Kompirmaen no brown spot (nagtimbukel a lesion nga addaan kolor dapo a tengnga) ket saan a blast (kasla diamante).",
            "Pasayaaten ti kinabaknang ti daga — mangikabil iti balanse a NPK ken organic matter sigun iti soil test.",
            "Ituntonan ti kurang a silicon ken potassium no nababa dagitoy iti soil test.",
            "Pagtalinaeden a nababaw ken patas ti irigasion; liklikan ti nabayag a kalgaw bayat ti tillering ken panagsabong.",
            "Ikkaten dagiti nakaro nga apektado a garami kalpasan ti panagapit; aradoen ti stubble.",
            "No napartak ti panagwaras, mangikabil iti mancozeb wenno propiconazole sigun iti label ken iti giya ti LGU.",
            "Iti sumaruno a panawen agusar iti certified ken fungicide-treated a bin-i manipud iti resistant wenno tolerant a barayti.",
        ],
        "cultural": [
            "Balanse nga abono — liklikan ti panangbisin iti mula sa mangikabil iti nabuslon a nitrogen iti udina.",
            "Nasayaat a panangtaming iti danum; kissayan ti stress kadagiti napateg a yugto ti panagdakkel.",
            "Panagsinnukat ti mula ken kinadalus ti talon no agsubsubli ti brown spot iti tunggal panawen.",
        ],
        "chemical": [
            "Agispray iti fungicide no agwaras dagiti marka kalpasan a naituntonan ti kinabaknang ti daga ken ti water stress.",
        ],
        "prevention": [
            "Ti nasalun-at ken nabaknang a daga ti kasayaatan a depensa — masansan a pagilasinan ti nutrient wenno moisture stress ti brown spot.",
            "Agusar iti nasayaat a bin-i ken liklikan ti agmula manipud iti bukel dagiti nakaro nga apektado a talon.",
        ],
        "resistant_varieties": [
            "Damagenyo iti PhilRice wenno iti MAO ti barayti nga addaan nasayaat a tolerance iti brown spot iti lugaryo."
        ],
        "farmer_tip": (
            "Ipangpangruna ti PhilRice ti panangituntonan iti kinabaknang ti daga ken iti water stress — saan a maisimpa "
            "ti fungicide a maymaysa ti nakapuy a daga."
        ),
    },
    "sheath blight": {
        "name": "Sheath Blight",
        "summary": (
            "Sakit a gapuanan ti fungus a papigsaen ti nangato a humidity, nairut a canopy, ken nalabes a nitrogen. "
            "Umuli dagiti lesion manipud iti leaf sheath ket mabalin a madadael ti panicle."
        ),
        "symptoms": [
            "Oval wenno saan a patas a greenish-gray a lesion nga addaan kolor kape nga igid iti leaf sheath",
            "Mabalin nga agdikket dagiti lesion ket maruket ti sheath; mabalin a matay dagiti akin-ngato a bulong",
            "Adda sclerotia (babassit a nangisit a pellet) kadagiti apektado a tisiu",
            "Ad-adda a nakaro kadagiti nairut ken nalabes ti abonona a talon",
        ],
        "procedure": [
            "Agsukimat kadagiti igid ti talon ken kadagiti nababa a paset a nangato ti humidity.",
            "Kissayan ti kaadu ti mula iti sumaruno — ti nalawlawa a nagbaetan ket mangpasayaat iti panagayus ti angin.",
            "Kissayan ti nitrogen; liklikan ti nalusaw ken nakapuy a puon.",
            "Pagpaawanen ti danum no maminsan no naunday unay a nakaurnong ti danum.",
            "Mangikabil iti validamycin, hexaconazole, wenno propiconazole iti umuna a lesion iti sheath.",
            "Saan nga usaren a mulch ti naspol a garami.",
        ],
        "cultural": [
            "Balanse nga abono ken maidasig a kaadu ti mula.",
            "Sagpaminsan a panagpagango kadagiti irrigated a talon iti panawen ti tudo.",
        ],
        "chemical": ["Agispray iti umuna a lesion iti akin-baba a sheath — saan a kalpasan a maapektaran ti flag leaf."],
        "prevention": ["Liklikan ti nalabes a nitrogen ken ti nairut a mula — dagitoy ti dua a kangrunaan a gapu ti sheath blight."],
        "farmer_tip": "Rugian ti panangtaming iti umuna a lesion iti sheath; naladaw unayen no apektado metten ti flag leaf.",
    },
    "sheath rot": {
        "name": "Sheath Rot",
        "summary": (
            "Maimpeksionan ti flag-leaf sheath ken ti panicle, isu nga agbaliw ti kolor wenno awan lamanna ti bukel. "
            "Masansan a sumrek iti sugat a naggapu kadagiti insekto wenno iti trabaho iti talon."
        ),
        "symptoms": [
            "Agbaliw ti kolor ken maruket ti flag-leaf sheath no asidegen ti heading",
            "Mabalin a saan a naan-anay a rummuar ti panicle wenno chalky ken nabaliwan ti kolor ti bukel",
            "Adda musty nga angotna kadagiti nakaro a kaso",
        ],
        "procedure": [
            "Agsukimat iti booting no adda panagbaliw ti kolor ti sheath sakbay a rummuar ti panicle.",
            "Kontrolen ti stem borer ken liklikan ti mekanikal a sugat bayat ti panagtrabaho.",
            "Siguraduen nga umanay ti potassium; liklikan ti saan a balanse a nitrogen.",
            "Mangikabil iti carbendazim wenno propiconazole iti booting no nakompirma ken adda balakad ti lokal nga opisina.",
            "Agusar iti nadalus a bin-i ken dadaelen ti apektado a garami.",
        ],
        "cultural": [
            "Balanse a potassium ken umno a kauneg ti danum iti heading.",
            "Integrated pest management maibusor iti stem borer a manglukat iti dalan ti impeksion.",
        ],
        "chemical": ["Maysa agingga dua nga ispray ti fungicide iti booting sakbay a rummuar ti panicle."],
        "prevention": ["Ti nasalun-at a panagruar ti panicle ket agpannuray iti panangsalaknib iti flag leaf manipud iti sugat ken stress."],
        "farmer_tip": "Masansan nga agkuyog ti sheath rot ken ti stem borer — kitaen dagiti abut ti borer iti isu met laeng a mula.",
    },
    "tungro": {
        "name": "Sakit a Tungro",
        "summary": (
            "Viral a sakit nga awiten ti green leafhopper (Nephotettix spp.). "
            "Awan agas para iti apektado a mula — ipangruna ti panangikkat ken ti panangkontrol iti insekto."
        ),
        "symptoms": [
            "Pandek a mula a duyaw wenno kahel ti bulongna, nangnangruna kadagiti ubbing a tiller",
            "Manmano ti tiller ken bassit wenno awan a pulos ti bukel",
            "Adda leafhopper iti talon (babassit a berde nga insekto nga aglagto no masinga)",
        ],
        "procedure": [
            "Ikkaten ken dadaelen a dagus dagiti apektado a mula — saan nga i-compost iti asideg ti talon.",
            "Kontrolen ti green leafhopper babaen ti naaprobaran nga insecticide no lumabes laeng iti threshold.",
            "Liklikan ti agsisinnublat a panagmula a mangtaginayon iti leafhopper ken virus iti intero a panawen.",
            "Agmula iti barayti a resistant iti tungro (kas iti NSIC Rc 120, Rc 222 no maidasig).",
            "Pagsasabayen ti petsa ti panagmula kadagiti kaarruba tapno mapugsat ti siklo ti sakit.",
            "Ikkaten dagiti ruot iti tambak a paggianan dagiti leafhopper.",
        ],
        "cultural": [
            "Agusar iti resistant a barayti kas kangrunaan nga estratehia — ibilang ti PhilRice ti tungro a peggad iti panawen ti tudo.",
            "Pababaan ti kabayag ti crop establishment (kas iti modified dapog) tapno umababa ti panawen ti pannakaexpose no mabalin.",
        ],
        "chemical": [
            "Awan ti pestisidio a mangagas iti tungro — agispray laeng tapno makontrol ti green leafhopper (IPM threshold).",
        ],
        "prevention": [
            "Saan nga agraep iti bunubon a naggapu iti pagbunubonan nga addaan tungro.",
            "Bantayan ti bilang dagiti leafhopper kada lawas iti umuna a 30 nga aldaw kalpasan ti panagraep.",
        ],
        "resistant_varieties": ["NSIC Rc 120 ken dadduma pay a barayti a resistant iti tungro para iti rehionyo."],
        "farmer_tip": "Ti nasapa a panangikkat kadagiti sumagmamano nga apektado a hill ket makaisalakan iti intero a talon — awan kemikal nga agas iti tungro.",
    },
    "leaf smut": {
        "name": "Leaf Smut",
        "summary": "Kadawyan a bassit laeng a sakit ti bulong ngem mabalin a makakissay iti photosynthesis no nakaro. Agwaras babaen dagiti spores iti bin-i ken iti angin.",
        "symptoms": [
            "Babassit a nangisit ken kasla pulbos a smut sori iti bulong (masansan iti nagbaetan dagiti urat)",
            "Adu a babassit a nangisit a tuldok iti bulong",
            "Manmano a mangpatay iti mula ngem mabalin nga agkapuy ti panagdakkel no nakaro",
        ],
        "procedure": [
            "Agusar iti certified a bin-i nga awan sakitna iti sumaruno a panagmula.",
            "Hot-water seed treatment (52–54 °C iti las-ud ti 10 a minuto) wenno naaprobaran a seed treatment sakbay ti panagimula.",
            "Ikkaten ken dadaelen dagiti nakaro nga apektado a bulong iti pagbunubonan ken iti kangrunaan a talon.",
            "Liklikan ti nalabes a nitrogen a pakaigapuan ti nalusaw ken nakapuy a panagdakkel.",
            "Agsukat iti mulmula kadagiti talon nga agsubsubli ti smut.",
        ],
        "cultural": ["Ti kinadalus ti talon ken ti kalidad ti bin-i ti kangrunaan a kontrol nga idasig ti PhilRice maibusor iti smut."],
        "chemical": ["Manmano a masapul ti ispray iti talon — ti seed treatment ti kangrunaan a kemikal a kontrol."],
        "prevention": ["Saan nga agmula iti bin-i a naggapu iti talon nga adu ti smutna."],
        "farmer_tip": "Masansan nga adda iti bin-i ti leaf smut — nadalus a bin-i ti solusion iti kaaduanna, saan a ti ispray iti talon.",
    },
    "narrow brown leaf spot": {
        "name": "Narrow Brown Leaf Spot",
        "summary": "Sakit ti bulong a gagangay iti pagay a kurang iti sustansia wenno danum, nangnangruna kadagiti dadduma nga upland a lugar.",
        "symptoms": [
            "Nailet ken ababa a guhit a kolor kape wenno marka nga addaan kolor dapo a tengnga, masansan iti urat",
            "Mabalin a kaas-asping ti brown spot ngem nailetlet ken naglinlinteg dagiti lesion",
            "Ad-adda a nakaro no nababa ti kinabaknang ti daga wenno adda moisture stress",
        ],
        "procedure": [
            "Pasayaaten ti kinabaknang ti daga sigun iti soil test — balanse a N, P, K.",
            "Pagtalinaeden ti tuloy-tuloy a moisture bayat ti tillering ken panicle initiation.",
            "Mangikabil iti mancozeb wenno propiconazole no agwarwaras latta dagiti lesion kalpasan a naituntonan ti kinabaknang ti daga.",
            "Agusar iti tolerant a barayti no adda.",
        ],
        "cultural": ["Ituntonan ti naggapuan ti stress — masansan a pagilasinan ti nakapuy a sustansia wenno danum ti NBLS."],
        "chemical": ["Agispray laeng kalpasan a naituntonan ti kinabaknang ti daga ken ti moisture stress."],
        "prevention": ["Ti nasalun-at a daga ken ti tuloy-tuloy a danum ti nabayag a solusion."],
        "farmer_tip": "Simpaen nga umuna ti sustansia ken ti danum; nakapuy ti resulta ti ispray no saan a naituntonan ti stress.",
    },
    "false smut": {
        "name": "False Smut",
        "summary": "Sukatanna ti bukel iti kahel agingga berde a smut ball iti heading. Papigsaen ti nangato a humidity ken ti nalabes a nitrogen.",
        "symptoms": [
            "Nalukneng a kahel agingga berde a bola iti bukel iti heading",
            "Bumtak ti smut ball ket agiwaras iti spores a makaakar iti sumaruno a panawen",
            "Ad-adda a mapasamak no nangato ti nitrogen no asidegen ti heading",
        ],
        "procedure": [
            "Urnongen ken dadaelen dagiti smut ball sakbay a bumtak — isupot ken puoran, saan nga ibati iti talon.",
            "Kissayan ti nitrogen no asidegen ti heading; bingayen ti aplikasion iti nasapsapa.",
            "Mangikabil iti propiconazole wenno tebuconazole iti booting agingga iti nasapa a heading no adda problema idi.",
            "Saan nga usaren a bin-i ti bukel manipud kadagiti apektado a panicle.",
            "Agsukat iti mula a saan a pagay no nakaro ti false smut.",
        ],
        "cultural": ["Saan a nalabes a nitrogen ken nasayaat a panagayus ti danum iti reproductive stage."],
        "chemical": ["Azole spray iti booting agingga iti nasapa a heading — sakbay nga agparang dagiti berde a smut ball."],
        "prevention": ["Ti certified seed ken ti panangdadael kadagiti smut ball ti kapraktikalan nga aramiden ti mannalon."],
        "farmer_tip": "Ti panangikkat kadagiti smut ball babaen ti ima sakbay ti panagapit ket ad-adda a makatulong iti sumaruno a mula ngem ti maudi a fungicide spray.",
    },
    "bakanae": {
        "name": "Bakanae (Foolish Seedling)",
        "summary": "Sakit nga awiten ti bin-i a mangpataud iti nalabes a kangato, nalusiaw, ken nakapuy a bunubon. Gagangay no bukod a bin-i ti maus-usar.",
        "symptoms": [
            "Atiddog, naingpis, ken nalusiaw a berde a bunubon iti pagbunubonan",
            "Mabalin a matay ti bunubon sakbay wenno kalpasan ti panagraep",
            "Awan lamanna wenno manmano ti bukel kadagiti nabati a mula",
        ],
        "procedure": [
            "Ilaksid ken dadaelen dagiti saan a gagangay a nangato a bunubon — saan nga imula.",
            "Agusar iti certified ken fungicide-treated a bin-i manipud iti mapagtalkan a paggapuan.",
            "Hot-water seed treatment (52–54 °C iti las-ud ti 10–15 a minuto) sakbay ti panagimula no adda bakanae idi.",
            "Carbendazim wenno thiram a seed treatment sigun iti label sakbay ti panagimula iti pagbunubonan.",
            "Liklikan ti agmula iti bin-i manipud iti talon nga addaan pakasaritaan ti bakanae.",
        ],
        "cultural": ["Kinadalus ti pagbunubonan ken nadalus a danum; liklikan ti nairut ken nakapuy a bunubon."],
        "chemical": ["Agasan ti bin-i sakbay ti pagbunubonan — awan epektibo a foliar nga agas iti kangrunaan a talon."],
        "prevention": ["Malapdan ti bakanae iti bin-i, saan a maagasan iti kangrunaan a talon."],
        "farmer_tip": "Maysa a nalusiaw ken nalabes ti kangatona a bunubon iti tray ket mabalin a pagilasinan ti naimpeksionan a lote ti bin-i — sukatan ti bin-i, saan laeng a parutan ti maysa.",
    },
    "hispa": {
        "name": "Rice Hispa",
        "summary": "Kudkuden ti adult ken ti larva ti bulong, mangibati iti puraw a guhit. Saan a sakit ngem nairaman iti giya iti panangsalaknib iti talon.",
        "symptoms": [
            "Puraw ken kasla papel a nakudkod a paset ti bulong (feeding marks)",
            "Nadadael a murdong ti bulong iti pagbunubonan",
            "Makita dagiti adult a beetle iti bulong",
        ],
        "procedure": [
            "Putden ken dadaelen dagiti nadadael a murdong ti bulong iti pagbunubonan bayat a nalag-an pay ti infestation.",
            "Agsukimat kada lawas; agasan laeng no lumabes iti lokal nga economic threshold.",
            "Mangikabil iti maidasig nga insecticide (kas iti chlorpyrifos, imidacloprid) sigun iti label no naabot ti threshold.",
            "Liklikan ti saan a masapul a broad-spectrum spray a mangpatay kadagiti natural enemies.",
            "Pagtalinaeden ti umno a nagbaetan para iti nasayaat a panagayus ti angin.",
        ],
        "cultural": ["Nasapa a panagputed iti pagbunubonan; salaknibanyo dagiti lawwalawwa ken dagiti parasitoid."],
        "chemical": ["Agispray laeng no ipakita ti scouting nga adda makadadael a populasion — surotenyo ti IPM threshold."],
        "prevention": ["Ad-adda a makaibtur dagiti nasalun-at ken awan stressna a mula iti nalag-an a pannangan ti hispa."],
        "farmer_tip": "Masansan a saan a kasapulan ti ispray iti nalag-an a pannangan — kitaen nga umuna ti beetle ken ti porsiento dagiti apektado a bulong.",
    },
    "dead heart": {
        "name": "Dead Heart (Stem Borer)",
        "summary": "Agabut ti larva iti puon; matay ti akin-tengnga a saringit ket nalaka a maparut. Kangrunaan a parikut nga insekto iti pagay ditoy Pilipinas.",
        "symptoms": [
            "Agduyaw ken matay ti akin-tengnga a tiller (dead heart) iti vegetative stage",
            "Puraw ken awan lamanna a panicle iti heading (white head) gapu iti panicle-feeding damage",
            "Babassit nga abut ken frass iti asideg ti puon ti tiller",
        ],
        "procedure": [
            "Parutan ken dadaelen dagiti dead-heart a tiller tapno matay ti larva iti unegna.",
            "Mangikabil iti pheromone trap tapno mabantayan dagiti kulibangbang ken maituon ti panangkontrol.",
            "Mangikabil iti granular nga insecticide (cartap, chlorantraniliprole) iti tillering sigun iti rekomendasion ti LGU.",
            "Liklikan ti ratoon crop kadagiti talon a nakaro ti infestation.",
            "Paaduen dagiti egg parasitoid babaen ti panangkissay iti saan a masapul nga insecticide spray.",
        ],
        "cultural": [
            "Agsasabay a panagmula, kinadalus ti talon, ken panangikkat kadagiti stubble a paggianan dagiti larva.",
        ],
        "chemical": ["Mangikabil iti granular nga insecticide iti tillering no lumabes iti threshold dagiti egg mass wenno dead heart."],
        "prevention": ["Mangrugi ti panangtaming iti stem borer iti tillering — ti white head no agapiten ket kayatna a sawen a nadadaelen sumagmamano a lawas ti napalabas."],
        "farmer_tip": "Maysa a dead heart a tiller ita ket mabalin nga agbalin nga adu a white head inton apit — ikkaten a nasapa.",
    },
    "physiological stress": {
        "name": "Bassit a Physiological Stress",
        "summary": (
            "Adda bassit a panagbaliw ti kolor ti bulong (natikag ken kolor kape a murdong, bassit a panagduyaw, wenno "
            "marka ti danum) ngem awan dagiti agwarwaras a lesion ti blast, brown spot, wenno bacterial blight. "
            "Gagangay a normal a senescence wenno nalag-an nga abiotic stress laeng daytoy — saan a kasapulan ti pestisidio."
        ),
        "symptoms": [
            "Natikag ken kolor kape a murdong iti berde pay a bulong (tip burn / senescence)",
            "Bassit ken patas a panagduyaw nga awan ti nalawag a marka, guhit, wenno kasla pulbos a tubo",
            "Marka ti danum wenno pitak a mapunas ken saan nga agwaras iti inaldaw",
            "Awan ti water-soaked nga igid, kasla spindle a lesion, wenno oval a marka a kolor kape",
        ],
        "procedure": [
            "SAANYO nga ispreyan iti fungicide wenno bactericide — awan ti pathogen a kontrolen.",
            "Kitaen ti panangtaming iti danum: liklikan ti agsisinnublat a kalgaw ken nauneg a layus.",
            "Kitaen ti sustansia: ti kurang a potassium ken ti salinity ket pakaigapuan ti leaf-tip burn; "
            "mangikabil iti abono sigun iti soil test wenno iti PhilRice Nutrient Manager.",
            "Sukimaten manen ti isu met laeng a mula kalpasan ti 3–5 nga aldaw: agtalinaed a kasta dagiti marka ti stress, "
            "idinto ta dumakdakkel, umad-adu, wenno aglalaok ti pudno a sakit.",
            "No agparang ken agwaras dagiti baro a marka wenno guhit, i-scan manen ti bulong wenno kumonsulta "
            "iti municipal agriculturist (MAO).",
        ],
        "cultural": [
            "Ti natibker nga irigasion ken ti balanse nga abono ti manglapped iti kaaduan a tip burn ken panagduyaw.",
        ],
        "prevention": [
            "Normal ti nalag-an a panagtikag ti murdong dagiti daan a bulong bayat a lumakay ti mula — "
            "bantayan, ngem saan nga ibilang a sakit.",
        ],
        "farmer_tip": (
            "Ti kolor kape a murdong iti berde a bulong ket masansan a mawaw wenno kurang a sustansia, saan nga impeksion — "
            "bantayan iti sumagmamano nga aldaw sakbay ti panaggastos iti kemikal."
        ),
    },
    "healthy": {
        "name": "Nasalun-at a Pagay",
        "summary": "Kasla nasalun-at ti bulong. Ituloy dagiti nasayaat nga aramid iti talon ken ti regular a panagsukimat.",
        "procedure": [
            "Pagtalinaeden ti umno a kauneg ti danum ken ti panagayus daytoy sigun iti yugto ti panagdakkel.",
            "Mangikabil iti abono sigun iti soil test ken iti giya ti PhilRice Nutrient Manager.",
            "Agsukimat iti mamindua iti maysa a lawas iti panawen ti tudo para kadagiti nasapa a pagilasinan ti BLB, blast, ken tungro.",
            "Pagtalinaeden a nadalus ti tambak ken ikkaten dagiti ruot a paggianan dagiti peste ken sakit.",
        ],
        "cultural": ["Ti balanse a sustansia ken ti umno nga oras ti panagmula ket mangkissay kadagiti saan a nanamnama nga outbreak iti udina ti panawen."],
        "prevention": ["Nalaklaka ti panangliklik ngem ti agas — bantayan dagiti talon dagiti kaarruba no nabasa ti tiempo."],
        "farmer_tip": "Ti nasalun-at ita ket saan a kayatna a sawen nga awan sakit inton bigat — ituloy ti panagsukimat agingga iti heading.",
    },
}
