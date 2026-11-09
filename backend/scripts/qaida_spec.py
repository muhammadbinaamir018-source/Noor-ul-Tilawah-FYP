"""What is printed on each Qaida scan, band by band, in reading order.

One entry per scan. `bands` lists every band qaida_grid.bands() finds on that
scan, top to bottom. Each band is a tuple (kind, items, mode):

  ("one",   None)            whole band is one clip, nothing to recite
                             (headings, Urdu notes)
  ("none",  None)            decoration only, no clip
  ("vsplit", n)              band holds n stacked clips (title over a note),
                             nothing to recite
  ("cells", [...], mode)     ruled cells, right to left; blank cells are
                             skipped automatically, "-" marks a decoration cell
  ("split", [...], mode)     unruled row cut at its widest gaps, right to left
  ("splitx", [...], mode)    same, with a decoration at each end to drop
  ("cols", [(x0, x1, text), ...], mode)
                             explicit pixel columns, right to left
  ("one", text, mode)        whole band is one recitable item
  ("tall", k, text, mode)    one item spanning the next k bands (an ayah
                             continued on the next line)
  ("cellsplit", [[items per filled cell], ...], mode)
                             ruled cells right to left, each cut into its own
                             items; ["-"] skips a cell (a label spanning rows)
  ("cellvsplit", [[top, bottom] per cell], mode, lines)
                             items stacked inside each cell; the cell has
                             `lines` text lines, extra ones (references) dropped
  ("colsplit", [(x0, x1, [items]), ...], mode)
                             explicit pixel columns, each cut into its items
  ("boxes", [(x0, y0, x1, y1, text, row_offset), ...], mode)
                             explicit pixel boxes (tables the grid finder
                             cannot read)
  ("lesson", "NN")           not a band: the rest of the scan continues takhti
                             NN (its audio starts over at clip 1)

A page may also give "rows": [(y0, y1), ...] when the cells contain their own
underlines (rasm-ul-khat) that would otherwise read as rules, and
"skip_clips": [...] for byte-identical duplicate files that have no box.
Item text may be a (text, [variants]) tuple.

Clips are assigned in this order; the audio is grouped by takhti (lesson),
so a lesson's clips continue from one scan to the next. The order was checked
against clip durations, exact duplicate clips, and Whisper transcripts of the
reference audio (see map_qaida.py --check).

Item text is what is printed. map_qaida.py turns it into expected text:
  letter-name  "ب"    -> letter name (با) plus variants
  names        "لج"   -> each letter's name, joined (لام جيم); how the
                         reference reciter reads murakkabat
  syllable / word     -> the printed text itself
"""

FATHA, KASRA, DAMMA, SUKUN = "َ", "ِ", "ُ", "ْ"
KHARA_ZABAR, KHARI_ZER, ULTA_PESH = "ٰ", "ٖ", "ٗ"

# The 30-letter grid used from takhti 3 on, right to left, row by row.
GRID = [
    list("ابتثجحخد"),
    list("ذرزسشصضط"),
    list("ظعغفقكلم"),
    list("نوهءىے"),
]


def grid(mark: str) -> list[tuple]:
    """The four letter rows of a harakat page, each letter carrying `mark`."""
    return [("cells", [c + mark for c in row], "syllable") for row in GRID]


def madd_grid(suffix: str, prefix_mark: str, last_row: list[str], deco: bool) -> list[tuple]:
    rows = [[c + prefix_mark + suffix for c in row] for row in GRID[:3]]
    last = [c + prefix_mark + suffix for c in last_row] + (["-"] if deco else [])
    return [("cells", r, "syllable") for r in rows] + [("cells", last, "syllable")]


def leen_grid(suffix: str) -> list[tuple]:
    """Takhti 5: every letter with fatha and a quiescent و or ی; the last row
    sits between two decorations."""
    rows = [[c + FATHA + suffix for c in row] for row in GRID[:3]]
    last = ["-"] + [c + FATHA + suffix for c in "نوهءي"] + ["-"]
    return [("cells", r, "syllable") for r in rows] + [("cells", last, "syllable")]


def waqf_boxes() -> list[tuple]:
    """Scan 37 is a two-column table whose rows differ per column; the grid
    finder only sees the outer frame, so the cells are pixel boxes. Each row
    is its symbol (narrow cell) then its Urdu explanation; the right column is
    read first."""
    right = [(453, 569), (572, 632), (635, 747), (749, 863), (865, 980), (983, 1052), (1054, 1171), (1173, 1230), (1233, 1347)]
    left = [(453, 639), (642, 700), (702, 763), (765, 826), (829, 1074), (1076, 1199), (1201, 1347)]
    out = []
    for i, (y0, y1) in enumerate(right):
        out += [(924, y0, 994, y1, None, i), (541, y0, 922, y1, None, i)]
    for j, (y0, y1) in enumerate(left):
        out += [(464, y0, 539, y1, None, 9 + j), (88, y0, 462, y1, None, 9 + j)]
    return out


H, M = ("one", None), ("one", None)  # heading / Urdu note: a clip, nothing to recite

PAGES = {
    # ---- Takhti 1, mufradat (continues from page 1, clips 23-88) ----------
    "p02": {"lesson": "01", "bands": [
        ("cells", list("فقكل"), "letter-name"),
        ("cells", list("منوه"), "letter-name"),
        ("cols", [(777, 1013, "ء"), (540, 777, "ى"), (303, 540, "ے")], "letter-name"),
        H,  # امتحان
        ("cells", list("تطثسشصحهخ"), "letter-name"),
        ("cells", list("قذزضظاعنب"), "letter-name"),
        ("cells", list("فغىتملد"), "letter-name"),
        H,  # مرکبات میں ملے ہوئے حروف کی شکلیں
        ("cells", list("ابتجدعف"), "letter-name"),
        ("cells", [None] * 7, None),   # joined forms, read with Urdu ("... کی شکلیں")
        ("cells", list("كلمنهءى"), "letter-name"),
        ("cells", [None] * 7, None),
    ]},
    # ---- Takhti 2, murakkabat (clips 1-162) --------------------------------
    "p03": {"lesson": "02", "bands": [
        H,
        ("split", list("ال"), "letter-name"),
        ("split", ["لا", "جا", "لا", "ما", "لا", "با"], "names"),
        ("split", ["لا", "لج", "لا", "لب", "لا", "بلب"], "names"),
        ("splitx", list("ككك"), "letter-name"),
        ("split", ["كب", "كت", "كا", "كا", "بكت", "تكث"], "names"),
        ("split", list("بتثنى"), "letter-name"),
        ("split", ["با", "تا", "ثا", "نا", "يا", "بس"], "names"),
        ("split", ["يس", "نس", "تس", "ثس", "ثج", "تح"], "names"),
    ]},
    "p04": {"lesson": "02", "bands": [
        ("split", ["نخ", "يح", "بج", "بم", "يم", "نم"], "names"),
        ("split", ["تم", "ثم", "بي", "يي", "ني", "تي"], "names"),
        ("split", ["ثي", "نبل", "تنل", "بيل", "يتل", "ثل"], "names"),
        ("split", ["نبن", "بنن", "تين", "يتن", "ثثن", "نين"], "names"),
        ("splitx", list("جحخ"), "letter-name"),
        ("split", ["حث", "خب", "جت", "تحت", "يجب"], "names"),
        ("splitx", ["صحب", "بخت"], "names"),
        ("splitx", ["ة", "ه", "ه"], "letter-name"),
        ("split", ["بة", "يه", "ته", "يهب", "بها", "بهه"], "names"),
    ]},
    "p05": {"lesson": "02", "bands": [
        ("splitx", list("دذرز"), "letter-name"),
        ("split", ["جد", "خذ", "جر", "خز", "ير", "تز"], "names"),
        ("split", ["مد", "كذ", "فر", "مز"], "names"),
        ("split", list("سشصضطظ"), "letter-name"),
        ("split", ["سل", "شل", "صب", "طب", "ضا"], "names"),
        ("split", ["سر", "شق", "ضي", "ظا"], "names"),
        ("splitx", list("عغء"), "letter-name"),
        ("split", ["عز", "غر", "صع", "ضغ", "بعد", "تغد"], "names"),
        # أ ؤ ئ clips explain in Urdu ("ہمزہ الف کی شکل میں"); only ء is recitable
        ("splitx", ["ء", None, None, None], "letter-name"),
    ]},
    "p06": {"lesson": "02", "bands": [
        ("splitx", list("فقوم"), "letter-name"),
        ("split", ["قو", "فو", "فقل", "قفل", "يف", "حم"], "names"),
        ("split", ["كم", "لم", "تم", "تمت"], "names"),
        H,  # امتحان
        ("split", ["اياك", "نستعين", "وما", "يدريك"], "word"),
        ("split", ["لعله", "يزكى", "او", "يذكر", "فتنفعه"], "word"),
        ("split", ["الذكرى", "فسيكفيهم", "الله", "قبلتهم"], "word"),
        ("split", ["مستقيم", "بحجارة", "من", "سجيل"], "word"),
        ("split", ["فجعلهم", "كعصف", "مأكول", "لتنبئن"], "word"),
        ("none", None),
    ]},
    # ---- Takhti 3, harakat (clips 1-138) -----------------------------------
    "p07": {"lesson": "03", "bands": [
        ("vsplit", 2),
        *grid(FATHA),
        M,  # مشق
        ("cells", ["دَرَسَ", "وَدَعَ", "عَبَدَ", "كَسَبَ"], "word"),
        ("cells", ["دَخَلَ", "بَلَغَ", "وَجَدَ", "سَجَدَ"], "word"),
        M,  # footnote
    ]},
    "p08": {"lesson": "03", "bands": [
        ("vsplit", 2),
        *grid(KASRA),
        M,
        ("cells", ["اِبِلِ", "رَدِفَ", "حَمِدَ", "شَهِدَ"], "word"),
        ("cells", ["بَخِلَ", "سَخِرَ", "رَحِمَ", "لَعِبَ"], "word"),
        M,
        ("cells", ["شَرِبَ", "عَمِلَ", "بَرِقَ", "خَطِفَ"], "word"),
        ("cells", ["جَزِعَ", "سَقِمَ", "غَشِيَ", "نَسِيَ"], "word"),
    ]},
    "p09": {"lesson": "03", "bands": [
        ("vsplit", 2),
        *grid(DAMMA),
        M,
        ("cells", ["رُسُلُ", "سُدُسُ", "فُقِدَ", "قُتِلَ"], "word"),
        ("cells", ["نُكِسَ", "كَرُمَ", "عُلِمَ", "هُدِيَ"], "word"),
        M,
        ("cells", ["صُحُفُ", "وُسِعَ", "قُدِرَ", "نُصِرَ"], "word"),
    ]},
    # ---- Takhti 4, huroof-e-madda (clips 1-277) ----------------------------
    "p10": {"lesson": "04", "bands": [
        H, M,
        ("cells", ["ءَا", "بَا", "تَا", "ثَا", "جَا", "حَا", "خَا", "دَا"], "syllable"),
        ("cells", ["ذَا", "رَا", "زَا", "سَا", "شَا", "صَا", "ضَا", "طَا"], "syllable"),
        ("cells", ["ظَا", "عَا", "غَا", "فَا", "قَا", "كَا", "لَا", "مَا"], "syllable"),
        ("cells", ["نَا", "وَا", "هَا", "يَا"], "syllable"),
        M,
        ("cells", ["زَادَ", "خَافَ", "تَابَ", "جَاهَدَ"], "word"),
        ("cells", ["حَاسَبَ", "جُنَاحُ", "شَارَبَ", "فَرَاغَ"], "word"),
        M,
        ("cells", ["خَادَعَ", "قَاتَلَ", "صَابَرَ", "تَعَالَ"], "word"),
    ]},
    "p11": {"lesson": "04", "bands": [
        ("cols", [(900, 1010, None), (70, 900, None)], None),  # ی heading, Urdu note
        *madd_grid("يْ", KASRA, list("نوهءي"), deco=True),
        M,
        ("cells", ["دِيْنِيْ", "فِيْهِ", "اَرِنِيْ", "كِتَابِيْ"], "word"),
        ("cells", ["اُجِيْبُ", "يُوَارِيْ", "مَفَاتِيْحُ", "رَازِقِيْنَ"], "word"),
        M,
        ("cells", ["عِبَادِيْ", "عَذَابِيْ", "تَمَاثِيْلُ", "مَقَادِيْرُ"], "word"),
        ("cells", ["اَخِيْهِ", "بَنِيْهِ"], "word"),
        M,
    ]},
    "p12": {"lesson": "04", "bands": [
        ("vsplit", 2),
        *madd_grid("وْ", DAMMA, list("نوهءي"), deco=True),
        M,
        ("cells", ["نُوْحُ", "طُوْرُ", "تُوْبُوْا", "نُوْرُ"], "word"),
        ("cells", ["قَالُوْا", "يُوْحِيْ", "يَقُوْمُ", "تَكُوْنُ"], "word"),
        M,
        ("cells", ["مُسْلِمُوْنَ", "قَارُوْنُ", "دَاخِرُوْنَ", "سَبَقُوْنَا"], "word"),
        ("cells", ["بَاسِطُوْنَ", "رَاجِعُوْنَ"], "word"),
    ]},
    "p13": {"lesson": "04", "bands": [
        ("vsplit", 2),
        *grid(KHARA_ZABAR),
        M,
        ("cells", ["اٰدَمَ", "اٰمَنَ", "مٰلِكِ", "اَبٰوْهُ"], "word"),
        ("cells", ["سَمٰوٰتِ", "غٰوِيْنَ", "يُصٰلِحُ", "اِلٰهُنَا"], "word"),
        M,
        ("cells", ["هٰذَا", "كِتٰبُ", "رِسٰلٰتِ", "ذٰلِكَ"], "word"),
    ]},
    "p14": {"lesson": "04", "bands": [
        ("vsplit", 2),
        *grid(KHARI_ZER),
        M,
        ("cells", ["اِلٰفِ", "بِهٖ", "عِبَادِهٖ", "رُسُلِهٖ"], "word"),
        ("cells", ["نُوْرِهٖ", "وَقِيْلِهٖ", "هٰذِهٖ", "بِكَلِمٰتِهٖ"], "word"),
        M,
        ("cells", ["بِاٰيٰتِهٖ", "بِكِتٰبِهٖ", "بِيَمِيْنِهٖ", "فِيْهِ"], "word"),
    ]},
    "p15": {"lesson": "04", "bands": [
        ("vsplit", 2),
        *grid(ULTA_PESH),
        M,
        ("cells", ["لَهٗ", "دَاوٗدَ", "رَسُوْلُهٗ", "اٰيٰتُهٗ"], "word"),
        ("cells", ["جُنُوْدُهٗ", "تِلَاوَتُهٗ", "وَرِثَهٗ", "مَوَازِيْنُهٗ"], "word"),
        M,
        ("cells", ["جَعَلَهٗ", "مَاوٗرِيَ", "غَاوٗنَ", "قَرِيْنُهٗ"], "word"),
    ]},
    # ---- Takhti 5, huroof-e-leen (clips 1-89) -------------------------------
    "p16": {"lesson": "05", "bands": [
        ("vsplit", 3),  # title, Urdu note, وْ
        *leen_grid("وْ"),
        M,  # مشق
        ("cells", ["اَوْفِ", "حَوْلَ", "صَوْمُ", "سَوْفَ"], "word"),
        ("cells", ["كَوْثَرَ", "شَرَوْهُ", "يَقَوْمُ", "مَوْءُدَةُ"], "word"),
        M,
    ]},
    "p17": {"lesson": "05", "bands": [
        ("vsplit", 2),  # Urdu note, ی
        *leen_grid("يْ"),
        M,
        ("cells", ["اَيْنَ", "صَيْفُ", "اَبَوَيْهِ", "يَلَيْتَنِيْ"], "word"),
        ("cells", ["اَوْحَيْتَ", "عَيْنَيْنِ", "لَارَيْبَ", "غَيْرِيْ"], "word"),
        ("split", ["بَيْنَ يَدَيْهِ", "يُنَادَوْنَ", "هَيْهَاتَ"], "phrase"),
        ("split", ["اَوْجَسَ", "سُلَيْمٰنُ", "فَتَعَالَيْنَ"], "phrase"),
        M,
    ]},
    # ---- Takhti 6, mashq (clips 1-25) ---------------------------------------
    "p18": {"lesson": "06", "bands": [
        H,
        ("split", ["بَ بْ بَاْ", "بِ بْ بِيْ", "بُ بْ بُوْ"], "phrase"),
        ("splitx", ["جُوْ جَوْ جِيْ جْ جَيْ"], "phrase"),
        ("split", ["خَلَقَ", "اِذَا وَقَبَ", "وَاِذَا قُرِئَ"], "phrase"),
        ("split", ["وَكَوَاعِبَ", "فِيْ جِيْدِهَا", "يَقُوْلُ"], "phrase"),
        ("split", ["فَعَقَرُوْهَا", "لَا يَمُوْتُ فِيْهَا", "مَاٰرِبُ"], "phrase"),
        ("split", ["يَوْمَ يَرَوْنَهَا", "وَصَاحِبَتِهٖ", "ذٰلِكَ"], "phrase"),
        ("split", ["فَقَالَ", "حٰفِظِيْنَ", "وَاُوْتِيَ كِتٰبَهٗ"], "phrase"),
        ("split", ["كَيْفَ فَعَلَ", "وَلِيْ دِيْنِ", "اَوْحٰى لَهَا"], "phrase"),
        ("split", ["بِمَا يُوْعُوْنَ", "وَطُوْرِ سِيْنِيْنَ"], "phrase"),
    ]},
    # ---- Takhti 7, tanween (clips 1-94) -------------------------------------
    "p19": {"lesson": "07", "bands": [
        H, M,
        ("colsplit", [(822, 1000, [None]), (70, 815, [c + "ًا" if c != "ا" else "اً" for c in "ابتثجحخدذ"])], "syllable"),
        ("split", [c + "ًا" for c in "رزسشصضطظعغ"], "syllable"),
        ("split", [c + "ًا" for c in "فقكلمنوهءي"], "syllable"),
        ("colsplit", [(822, 1000, [None]), (70, 815, [c + "ٍ" for c in "ابتثجحخدذر"])], "syllable"),
        ("split", [c + "ٍ" for c in "زسشصضطظعغف"], "syllable"),
        ("split", [c + "ٍ" for c in "قكلمنوهءيے"], "syllable"),
        ("colsplit", [(822, 1000, [None]), (70, 815, [c + "ٌ" for c in "ابتثجحخدذر"])], "syllable"),
        ("split", [c + "ٌ" for c in "زسشصضطظعغف"], "syllable"),
        ("split", [c + "ٌ" for c in "قكلمنوهءيے"], "syllable"),
    ]},
    # ---- Takhti 8, tanween and noon sakin (clips 1-39) ----------------------
    "p20": {"lesson": "08", "bands": [
        H, M,
        ("cellsplit", [["تًا", "تَنْ"], ["تٍ", "تِنْ"], ["تٌ", "تُنْ"]], "syllable"),
        ("cellsplit", [["ثًا", "ثَنْ"], ["ثٍ", "ثِنْ"], ["ثٌ", "ثُنْ"]], "syllable"),
        ("cellsplit", [["جًا", "جَنْ"], ["جٍ", "جِنْ"], ["جٌ", "جُنْ"]], "syllable"),
        ("cellsplit", [["دًا", "دَنْ"], ["دٍ", "دِنْ"], ["دٌ", "دُنْ"]], "syllable"),
        M,  # مشق
        ("cells", ["عَادًا", "سَوْءٍ", "قَرِيْبٌ", "فَمَنْ"], "word"),
        ("cells", ["عَظِيْمٌ", "كُنْ", "مَرَضٌ", "رَسُوْلٌ"], "word"),
        ("cells", ["شَيْئًا", "غِشَاوَةٌ", "غَاسِقٍ", "هُدًى"], "word"),
    ]},
    # ---- Takhti 9, izhar (clips 1-17) ---------------------------------------
    "p21": {"lesson": "09", "bands": [
        H, ("vsplit", 2),
        ("cells", ["طَيْرًا اَبَابِيْلَ", "نَارٌ حَامِيَةٌ"], "phrase"),
        ("cells", ["لِمَنْ خَشِيَ", "مِنْ غَيْرِهٖ"], "phrase"),
        ("cells", ["يَوْمَئِذٍ عَلَيْهَا", "مِنْهُ خِطَابًا"], "phrase"),
        ("cells", ["مِنْ عَيْنٍ اٰنِيَةٍ", "كُفُوًا اَحَدٌ"], "phrase"),
        ("none", None),
        ("split", ["فَمَنْ عُفِيَ", "مِنْ اَخِيْهِ", "مَنْ اَذِنَ"], "phrase"),
        ("split", ["كِتَابٌ حَكِيْمٌ", "عَذَابٌ غَلِيْظٌ"], "phrase"),
        ("splitx", ["حَاسِدٍ اِذَا حَسَدَ"], "phrase"),
    ]},
    # ---- Takhti 10, ikhfa (clips 1-18) --------------------------------------
    "p22": {"lesson": "10", "bands": [
        H, ("vsplit", 2),
        ("cells", ["اَنْتَ مُنْذِرُ", "كِرَامًا كَاتِبِيْنَ"], "phrase"),
        ("cells", ["كُنْتُ تُرٰبًا", "يَتِيْمًا فَاٰوٰى"], "phrase"),
        ("cells", ["مَنْ دَخَلَهٗ", "رَسُوْلٍ كَرِيْمٍ"], "phrase"),
        ("cells", ["نَارًا ذَاتَ", "كُتُبٌ فِيْهَا"], "phrase"),
        ("split", ["يَوْمَ يُنْفَخُ", "مَنْ طَغٰى", "فَتْحٌ قَرِيْبٌ"], "phrase"),
        ("split", ["مَنْ ثَقُلَتْ", "اَنْزَلْنَا", "شَيْءٍ قَدِيْرٌ"], "phrase"),
        M,
    ]},
    # ---- Takhti 11, jazm/sukoon (clips 1-42) --------------------------------
    "p23": {"lesson": "11", "bands": [
        H, M,
        ("cells", ["اَبْ", "اِبْ", "اُبْ", "جَبْ", "جِبْ", "جُبْ"], "syllable"),
        ("cells", ["بَجْ", "بِجْ", "بُجْ", "سَدْ", "سِدْ", "سُدْ"], "syllable"),
        ("cells", ["قَطْ", "قِطْ", "قُطْ", "جَقْ", "جِقْ", "جُقْ"], "syllable"),
        H,  # امتحان
        ("cells", ["بَسْ", "كِتْ", "جِعْ", "صَدْ", "غُلْ", "تَجْ"], "syllable"),
        ("cells", ["وَحْ", "سَبْ", "اِهْ", "نَصْ", "لَقْ", "جُلْ"], "syllable"),
        M,
        ("split", ["يَجْعَلْ", "لَقَدْ", "خَلَقْنَا", "اَطْعَمَنَا"], "word"),
        ("split", ["نَقْعًا", "بُرُوْجٍ", "مُحِيْطٌ", "اِهْدِنَا"], "word"),
    ]},
    # ---- Takhti 12, mashq sukoon (clips 1-25, 020 twice) --------------------
    "p24": {"lesson": "12", "bands": [
        H, M,
        ("cells", ["تَجْرِيْ", "تَقْوٰى", "يُغْنِيْ", "نُشِرَتْ"], "word"),
        ("cells", ["رُفِعَتْ", "يَنْقَلِبُ", "لَيْسَ", "حُشِرَتْ"], "word"),
        ("cells", ["يَعْلَمُ", "سُطِحَتْ", "يُوَسْوِسُ", "يُسْقَوْنَ"], "word"),
        ("split", ["وَاَلْقَتْ", "اَفَلَا يَعْلَمُ", "اِذَا بُعْثِرَ"], "phrase"),
        ("one", "فَاَلْهَمَهَا فُجُوْرَهَا وَتَقْوٰهَا", "phrase"),
        ("split", ["اَلَمْ نَشْرَحْ لَكَ صَدْرَكَ", "هَلْ"], "phrase"),
        ("split", ["اَتٰكَ", "فَلَهُمْ اَجْرٌ غَيْرُ مَمْنُوْنٍ"], "phrase"),
        ("split", ["اِقْرَأْ", "يُؤْتُوْنَ", "جِئْنَا", "يَأْمُرُوْنَ"], "word"),
    ]},
    # ---- Takhti 13, tashdeed (clips 1-54) -----------------------------------
    "p25": {"lesson": "13", "bands": [
        H, M,
        ("cells", ["اَبَّ", "اِبَّ", "اُبَّ", "اَبِّ", "اِبِّ", "اُبِّ"], "syllable"),
        ("cells", ["اَتَّ", "اِتَّ", "اُتَّ", "اَتِّ", "اِتِّ", "اُتِّ"], "syllable"),
        ("cells", ["اَجَّ", "اِجَّ", "اُجَّ", "اَجِّ", "اِجِّ", "اُجِّ"], "syllable"),
        ("cells", ["اَبًّا", "اِبًّا", "اُبًّا", "اَبٍّ", "اِبٍّ", "اُبٍّ"], "syllable"),
        ("cells", ["بَبٌّ", "بِبٌّ", "بُبٌّ", "تَبًّا", "تِبٍّ", "تُبٌّ"], "syllable"),
        ("cells", ["جَبَّ", "جِبَّ", "جُبَّ", "جَبِّ", "جِبِّ", "جُبِّ"], "syllable"),
        ("cells", ["فَعَّ", "فَعًّا", "كُوَّ", "تَوَّا", "سُجِّ", "عَدَّ"], "syllable"),
        ("cells", ["حُضُّ", "حَيَّ", "عَلَىَّ", "اِيَّاكَ", "يَظُنُّ", "ثُمَّ"], "word"),
        ("cells", ["-", "سُيِّرَتْ", "سُعِّرَتْ", "عُطِّلَتْ", "-"], "word"),
        M,
        ("none", None),
    ]},
    # ---- Takhti 14, ra sakina and mushaddada (clips 1-28) -------------------
    "p26": {"lesson": "14", "bands": [
        H, M, M,
        ("split", ["اَرْسَلْنَا", "قُرْاٰنٌ", "يَغُرَّكَ", "رُبَمَا"], "word"),
        ("split", ["رَءُوْفٌ", "مُسْتَقِرٌّ", "مَرَّةً", "بِرٌّ"], "word"),
        ("split", ["رَبِّ ارْجِعُوْنِ", "اَمِ ارْتَابُوْا", "اِرْجِعِيْ"], "phrase"),
        ("split", ["فِرْقَةٌ", "مِرْصَادًا", "قِرْطَاسٍ"], "word"),
        M,
        ("split", ["اُمِرْتُ", "مُنْهَمِرٍ", "وَاصْبِرْ"], "word"),
        ("split", ["مِنْ شَرِّ", "بَرٌّ", "بِرٌّ", "دُرِّيٌّ"], "word"),
        ("split", ["مُسْتَمِرٍّ", "ذُرِّيَّةً", "تُحَرِّمُ"], "word"),
    ]},
    # ---- Takhti 15 (qamari/shamsi) and 16 (lafz Allah) on one scan ----------
    "p27": {"lesson": "15", "bands": [
        H, M,
        ("cellsplit", [[None], ["وَالْقَمَرِ", "وَالْيَتٰمٰى", "وَالْاَقْرَبِيْنَ"]], "phrase"),
        ("cellsplit", [["-"], ["مَا الْقَارِعَةُ", "هُمُ الْمُفْلِحُوْنَ"]], "phrase"),
        ("cellsplit", [[None], ["اِذَا الشَّمْسُ", "كَانَ النَّاسُ"]], "phrase"),
        ("cellsplit", [["-"], ["يَقُوْلُ الرَّسُوْلُ", "اَقِيْمُوا الصَّلٰوةَ"]], "phrase"),
        ("lesson", "16"),
        H, M,
        ("cellsplit", [[None], ["اِنَّ اللّٰهَ", "قَالَ اللّٰهُ", "سَمِعَ اللّٰهُ"]], "phrase"),
        ("cellsplit", [[None], ["حُدُوْدُ اللّٰهِ", "يُرِيْدُ اللّٰهُ", "خَلْقُ اللّٰهِ"]], "phrase"),
        ("cellsplit", [[None], ["بَلِ اللّٰهُ", "دِيْنِ اللّٰهِ", "اَمْرِ اللّٰهِ"]], "phrase"),
    ]},
    # ---- Takhti 17, waqf (clips 1-11, some in two halves) -------------------
    "p28": {"lesson": "17", "bands": [
        H, M,
        ("one", "اَعُوْذُ بِاللّٰهِ مِنَ الشَّيْطٰنِ الرَّجِيْمِ", "phrase"),
        ("one", "بِسْمِ اللّٰهِ الرَّحْمٰنِ الرَّحِيْمِ", "phrase"),
        ("split", ["اَلْحَمْدُ لِلّٰهِ رَبِّ الْعٰلَمِيْنَ", "اِيَّاكَ"], "phrase"),
        ("split", ["نَعْبُدُ وَاِيَّاكَ نَسْتَعِيْنُ", "قُلْ هُوَ"], "phrase"),
        ("split", ["اللّٰهُ اَحَدٌ", "اِنَّ مَعَ الْعُسْرِ يُسْرًا"], "phrase"),
        ("split", ["فِيْهَا عَيْنٌ جَارِيَةٌ", "تُسْقٰى مِنْ"], "phrase"),
        ("split", ["عَيْنٍ اٰنِيَةٍ", "لَا تَسْمَعُ فِيْهَا لَاغِيَةً"], "phrase"),
    ]},
    # ---- Takhti 18 (tashdeed ma tashdeed) and 19 (madd) ---------------------
    "p29": {"lesson": "18", "bands": [
        H,
        ("cells", ["عِلِّيُّوْنَ", "يَزَّكّٰى", "يَذَّكَّرُ"], "word"),
        ("cells", ["مُزَّمِّلُ", "مُدَّثِّرُ", "عِلِّيِّيْنَ"], "word"),
        ("lesson", "19"),
        H, ("vsplit", 3),
        ("cellsplit", [[None], ["جَآءَ", "سِيْٓئَتْ", "سُوْٓءٌ"]], "word"),
        ("cellsplit", [[None], ["فِيْٓ اَمْرِنَا", "لَآ اُقْسِمُ", "قَالُوْٓا اٰنَّا"]], "phrase"),
        ("cellsplit", [[None], ["آلْئٰنَ", "حَآجُّوْكَ", "وَالصّٰٓفّٰتِ"]], "word"),
        ("none", None),
    ]},
    # ---- Takhti 20 (tashdeed after madd) and 21 (huroof muqatta'at) ---------
    "p30": {"lesson": "20", "skip_clips": ["a20006.mp3"], "bands": [
        H,
        ("one", "غَيْرِ الْمَغْضُوْبِ عَلَيْهِمْ وَلَا الضَّآلِّيْنَ", "phrase"),
        ("split", ["وَوَجَدَكَ ضَآلًّا فَهَدٰى", "فَاِذَا"], "phrase"),
        ("split", ["جَآءَتِ الطَّآمَّةُ الْكُبْرٰى", "فَاِذَا"], "phrase"),
        ("split", ["جَآءَتِ الصَّآخَّةُ", "وَلَا تَحٰٓضُّوْنَ عَلٰى"], "phrase"),
        ("split", ["طَعَامِ الْمِسْكِيْنِ", "وَاِذَا رَاَوْهُمْ"], "phrase"),
        ("one", "قَالُوْٓا اِنَّ هٰٓؤُلَآءِ لَضَآلُّوْنَ", "phrase"),
        ("lesson", "21"),
        H,
        ("cells", [("الف لام ميم", ["الٓمّٓ"]), ("الف لام ميم صاد", ["الٓمّٓصٓ"]), ("الف لام را", ["الٓرٰ"]),
                   ("الف لام ميم را", ["الٓمّٓرٰ"]), ("كاف ها يا عين صاد", ["كٓهٰيٰعٓصٓ"])], "phrase"),
        ("cells", [("طا ها", ["طٰهٰ"]), ("طا سين ميم", ["طٰسٓمّٓ"]), ("طا سين", ["طٰسٓ"]),
                   ("يا سين", ["يٰسٓ"]), ("صاد", ["صٓ"])], "phrase"),
        ("cells", [("حا ميم", ["حٰمٓ"]), ("حا ميم عين سين قاف", ["حٰمٓ عٓسٓقٓ"]),
                   ("قاف", ["قٓ"]), ("نون", ["نٓ"])], "phrase"),
    ]},
    # ---- Takhti 22, meem sakin (clips 1-20) ---------------------------------
    "p31": {"lesson": "22", "bands": [
        H, ("vsplit", 3), H,
        ("split", ["اِلَيْكُمْ مُّرْسَلُوْنَ", "لَهُمْ مَّا يَشَآءُوْنَ"], "phrase"),
        ("split", ["فِيْ قُلُوْبِهِمْ مَّرَضٌ", "فَهُمْ مُّعْرِضُوْنَ"], "phrase"),
        H,
        ("split", ["اِنَّ رَبَّهُمْ بِهِمْ", "تَرْمِيْهِمْ بِحِجَارَةٍ"], "phrase"),
        ("split", ["وَمَا هُمْ بِمُؤْمِنِيْنَ", "فَبَشِّرْهُمْ بِعَذَابٍ"], "phrase"),
        H,
        ("split", ["هُمْ فِيْهَا", "لَكُمْ دِيْنُكُمْ", "لَمْ يَلْبَسُوْا"], "phrase"),
        ("split", ["اَمْ لَمْ تُنْذِرْهُمْ", "لَهُمْ اَجْرٌ"], "phrase"),
    ]},
    # ---- Takhti 23, iqlab (clips 1-16) --------------------------------------
    "p32": {"lesson": "23", "bands": [
        H, ("vsplit", 2),
        ("split", ["مَنْ بَخِلَ", "لَنَسْفَعًا بِالنَّاصِيَةِ"], "phrase"),
        ("split", ["مِنْ بَيْنِ الصُّلْبِ", "كِرَامٍ بَرَرَةٍ"], "phrase"),
        ("split", ["مِنْ بَعْدِ", "مُطَهَّرَةٍ بِاَيْدِيْ سَفَرَةٍ"], "phrase"),
        ("split", ["بِذَنْبِهِمْ", "خَبِيْرٌ بِمَا تَعْمَلُوْنَ"], "phrase"),
        ("split", ["سَمِيْعٌ بَصِيْرٌ", "كَلَّا لَيُنْبَذَنَّ"], "phrase"),
        ("split", ["عَلِيْمٌ بِذَاتِ الصُّدُوْرِ", "رَجْعٌ بَعِيْدٌ"], "phrase"),
        ("splitx", ["رَسُوْلٌ بِمَا لَا تَهْوٰى"], "phrase"),
    ]},
    # ---- Takhti 24, idgham yarmaloon (clips 1-44, two scans) ----------------
    "p33": {"lesson": "24", "bands": [
        ("vsplit", 4),  # title, subheading, note, "الف"
        H,  # ادغام ر
        ("split", ["مُحَمَّدٌ رَّسُوْلُ اللّٰهِ", "مِنْ رَّبِّكَ"], "phrase"),
        ("split", ["غَفُوْرًا رَّحِيْمًا", "عِيْشَةٍ رَّاضِيَةٍ"], "phrase"),
        H,  # ادغام ل
        ("split", ["مِنْ لَّدُنْهُ", "كُلٌّ لَّهٗ", "يَكُنْ لَّهٗ"], "phrase"),
        ("split", ["رِزْقًا لَّكُمْ", "اُفٍّ لَّكُمْ", "مِنْ لَّبَنٍ"], "phrase"),
        ("vsplit", 2),  # "ب", note
    ]},
    "p34": {"lesson": "24", "bands": [
        H,  # ادغام ی
        ("split", ["خَيْرًا يَّرَهٗ", "لِقَوْمٍ يُّوْقِنُوْنَ", "مِنْ يَّوْمِ"], "phrase"),
        ("split", ["مَنْ يَّقُوْلُ", "وُجُوْهٌ يَّوْمَئِذٍ", "اَنْ يَّشَآءَ"], "phrase"),
        H,  # ادغام و
        ("split", ["اِلٰهًا وَّاحِدًا", "رَحِيْمٌ وَّدُوْدٌ", "اِنْ وَّهَبَتْ"], "phrase"),
        ("split", ["مِنْ وَّرَآئِهِمْ", "جَنّٰتٍ وَّعُيُوْنٍ", "مَنْ وُّعِدَ"], "phrase"),
        H,  # ادغام م
        ("split", ["رَسُوْلٌ مِّنَ اللّٰهِ", "صِرَاطًا مُّسْتَقِيْمًا"], "phrase"),
        ("split", ["رَحِيْقٍ مَّخْتُوْمٍ", "مِنْ مِّثْلِهٖ"], "phrase"),
        H,  # ادغام ن
        ("split", ["مِنْ نَّبِيٍّ", "نُوْرًا نَّهْدِيْ", "عَامِلَةٌ نَّاصِبَةٌ"], "phrase"),
        ("split", ["لَمَنْ نَّزِيْدُ", "فَمَنْ نَّكَثَ", "يَوْمَئِذٍ نَّاضِرَةٌ"], "phrase"),
    ]},
    # ---- Takhti 25, rasm-ul-khat (clips 1-60): written form over read form --
    "p35": {"lesson": "25", "rows": [(89, 206), (210, 499), (503, 760), (764, 1021), (1025, 1279), (1283, 1541)], "bands": [
        H, M,
        ("cellvsplit", [["اَفَئِنْ مَّاتَ"] * 2, ["لَاِلَى اللّٰهِ"] * 2, ["اَنْ تَبُوْٓءَ"] * 2], "phrase", 3),
        ("cellvsplit", [["مِنْ نَّبَاِءِ"] * 2, ["مَلَئِهٖ"] * 2, ["وَلَاَوْضَعُوْا"] * 2], "phrase", 3),
        ("cellvsplit", [["ثَمُوْدَ"] * 2, ["لِتَتْلُوَ"] * 2, ["لَنْ نَّدْعُوَ"] * 2], "phrase", 3),
        ("cellvsplit", [["لِشَيْءٍ"] * 2, ["لٰكِنْ"] * 2, ["لَاَذْبَحَنَّهٗ"] * 2], "phrase", 3),
    ]},
    "p36": {"lesson": "25", "rows": [(66, 322), (326, 583), (587, 846), (850, 1101), (1104, 1369), (1372, 1541)], "bands": [
        ("cellvsplit", [["لِيَرْبُوَ"] * 2, ["لَاِلَى الْجَحِيْمِ"] * 2, ["لِيَبْلُوَ"] * 2], "phrase", 3),
        ("cellvsplit", [["نَبْلُوَ"] * 2, ["بِئْسَ لِسْمُ"] * 2, ["لَاَنْتُمْ اَشَدُّ"] * 2], "phrase", 3),
        ("cellvsplit", [["سَلَاسِلَ"] * 2, ["قَوَارِيْرَ"] * 2, ["مَلَئِهِمْ"] * 2], "phrase", 3),
        ("cellvsplit", [["وَلَاَنَ عَابِدٌ مَّا عَبَدْتُّمْ"] * 2, ["مُوْسَا"] * 2, ["عِيْسَا"] * 2], "phrase", 3),
        ("cellvsplit", [["زَكَاتٍ"] * 2, ["صَلَاتٍ"] * 2, ["حَيَاتٍ"] * 2, ["مِشْكَاتٍ"] * 2], "word", 3),
        ("vsplit", 2),
    ]},
    # ---- Takhti 26, waqf signs (clips 1-37): a two-column table -------------
    "p37": {"lesson": "26", "bands": [
        H, M,
        ("none", None),  # table header
        ("boxes", waqf_boxes(), None),
        # footnote: (۱) on the right of line 1, (۲) runs from line 1 onto line 2
        ("boxes", [(560, 1360, 1010, 1440, None, 0), (88, 1360, 560, 1440, None, 0), (88, 1440, 1010, 1520, None, 1)], None),
    ]},
    # ---- Takhti 27, applying the rules (clips 1-26, two scans) --------------
    "p38": {"lesson": "27", "bands": [
        H,
        ("split", ["جَزَآءً", "حَدَآئِقَ", "مَلٰٓئِكَةُ", "اُولٰٓئِكَ"], "word"),
        ("split", ["اِنَّآ اَعْطَيْنٰكَ الْكَوْثَرَ", "وَكُلُّ اَمْرٍ"], "phrase"),
        ("split", ["مُّسْتَقِرٌّ", "اَلَّذِيْنَ هُمْ يُرَآءُوْنَ"], "phrase"),
        ("tall", 2, "تَنَزَّلُ الْمَلٰٓئِكَةُ وَالرُّوْحُ فِيْهَا بِاِذْنِ رَبِّهِمْ مِّنْ كُلِّ اَمْرٍ", "phrase"),
        ("one", "سَلٰمٌ هِيَ حَتّٰى مَطْلَعِ الْفَجْرِ", "phrase"),
        ("split", ["هُمَزَةٍ لُّمَزَةٍ الَّذِيْ", "اِهْدِنَا"], "phrase"),
        ("split", ["الصِّرَاطَ الْمُسْتَقِيْمَ", "لَا تَحْمِلُ"], "phrase"),
        ("split", ["رِزْقَهَا اللّٰهُ يَرْزُقُهَا", "وَلَاَجْرُ"], "phrase"),
    ]},
    "p39": {"lesson": "27", "bands": [
        ("split", ["الْاٰخِرَةِ اَكْبَرُ", "لَوْ كَانُوْا يَعْلَمُوْنَ"], "phrase"),
        ("split", ["فَذَكِّرْ", "اِنَّمَآ اَنْتَ مُذَكِّرٌ", "وَ"], "phrase"),
        ("split", ["لِيَتَمَتَّعُوْا فَسَوْفَ يَعْلَمُوْنَ", "وَ"], "phrase"),
        ("one", "لَمْ يَجْعَلْ لَّهٗ عِوَجًا قَيِّمًا لِّيُنْذِرَ", "phrase"),
        ("split", ["كَلَّا بَلْ رَانَ", "وَقِيْلَ مَنْ رَاقٍ"], "phrase"),
        ("split", ["اَلَسْتُ بِرَبِّكُمْ", "قَالُوْا بَلٰى شَهِدْنَا"], "phrase"),
        ("one", "اَنْ تَقُوْلُوْا", "phrase"),
        ("one", "اِرْجِعِيْٓ اِلٰى رَبِّكِ رَاضِيَةً مَّرْضِيَّةً", "phrase"),
        ("split", ["فَادْخُلِيْ فِيْ عِبٰدِيْ", "وَادْخُلِيْ جَنَّتِيْ"], "phrase"),
        ("none", None),
    ]},
}
