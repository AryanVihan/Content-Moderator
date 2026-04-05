"""Synthetic content data generator with fixed seed for reproducibility."""
from __future__ import annotations

import random
from datetime import datetime, timedelta
from typing import List

from server.models import ContentItem, UserHistory

# ---------------------------------------------------------------------------
# Fixed seed — all data is deterministic
# ---------------------------------------------------------------------------
RNG = random.Random(42)

# ---------------------------------------------------------------------------
# Template banks
# ---------------------------------------------------------------------------

HATE_SPEECH_TEMPLATES = [
    "People like [GROUP_A] should not be allowed in our [PLACE].",
    "I hate [GROUP_A], they are ruining [COUNTRY].",
    "[GROUP_A] are all criminals and need to be removed from [CITY].",
    "Nobody wants [GROUP_A] in their neighbourhood. Get out!",
    "Why do we keep letting [GROUP_A] into [COUNTRY]? They destroy everything.",
    "[GROUP_A] don't deserve the same rights as the rest of us.",
    "If you see a [GROUP_A] near you, make sure they know they're not welcome.",
    "The problem with [COUNTRY] is too many [GROUP_A] living here.",
    "[GROUP_A] are subhuman and should be treated accordingly.",
    "[GROUP_A] people are the source of all crime in [CITY].",
    "We need to purge [GROUP_A] from positions of power immediately.",
    "[GROUP_A] are parasites feeding off the hard work of real citizens.",
]

SAFE_CONTENT_TEMPLATES = [
    "Just watched [MOVIE] and it was absolutely amazing! Highly recommend.",
    "Does anyone know a good [FOOD] restaurant near [CITY]?",
    "Happy birthday to my amazing mom! You're the best [FAMILY_MEMBER] ever!",
    "Finally finished my [HOBBY] project after three weeks. So proud of myself!",
    "The weather in [CITY] today is gorgeous. Perfect day for a walk.",
    "Anyone else obsessed with the new season of [SHOW]? No spoilers please!",
    "Just got back from [CITY] — the food scene there is incredible.",
    "Looking for recommendations for a good [BOOK_GENRE] book to read this weekend.",
    "Shoutout to my local [BUSINESS_TYPE] for always going above and beyond.",
    "Can't believe how fast kids grow. My child started [SCHOOL_ACTIVITY] today!",
    "Tried making [FOOD] for the first time and it turned out great!",
    "The [SPORTS_TEAM] game last night was intense. What a comeback!",
    "Anyone else find [HOBBY] incredibly relaxing after a stressful day?",
    "Just adopted a rescue [PET] — meet the newest member of our family!",
    "My [FAMILY_MEMBER] recommended this [PRODUCT_TYPE] and it changed my life.",
]

MISINFORMATION_TEMPLATES = [
    "BREAKING: [FALSE_CLAIM] — Share before they delete this!",
    "Doctors don't want you to know that [HEALTH_CLAIM].",
    "URGENT: [FALSE_CLAIM]. The government is hiding this from you.",
    "Scientists have proven that [HEALTH_CLAIM] but big pharma won't allow it.",
    "This video proves [FALSE_CLAIM]. They're trying to censor this!",
    "My [FAMILY_MEMBER] works in [INSTITUTION] and confirmed [FALSE_CLAIM].",
    "[PRODUCT_TYPE] companies don't want you to know [HEALTH_CLAIM].",
    "SHARE IMMEDIATELY: [FALSE_CLAIM]. This is being suppressed!",
    "The real reason [INSTITUTION] doesn't talk about [HEALTH_CLAIM].",
    "A leaked document shows [FALSE_CLAIM]. Read before it's taken down.",
    "Natural remedy: [HEALTH_CLAIM]. Big pharma hates this one trick!",
    "WARNING: [FALSE_CLAIM]. Your [FAMILY_MEMBER] needs to know this now.",
]

SPAM_TEMPLATES = [
    "Earn $500/day from home! Click [FAKE_LINK] to learn how. Limited spots!",
    "Congratulations! You've won a free [PRODUCT_TYPE]. Claim at [FAKE_LINK].",
    "Buy cheap [PRODUCT_TYPE] — 90% off today only! [FAKE_LINK]",
    "Work from home opportunity! No experience needed. [FAKE_LINK]",
    "FREE [PRODUCT_TYPE] giveaway! Just follow and share. First 100 only!",
    "I made $10,000 last month doing this ONE thing. [FAKE_LINK]",
    "Replica [LUXURY_BRAND] available. DM for price list. Worldwide shipping.",
    "Click here to get a free [PRODUCT_TYPE]: [FAKE_LINK] (expires soon!)",
    "Make money fast! Join my team and earn passive income. [FAKE_LINK]",
    "Selling [PRODUCT_TYPE] — cheap prices, fast delivery. Message me!",
]

VIOLENCE_TEMPLATES = [
    "I'm going to [VIOLENT_ACT] anyone who [TRIGGER_CONDITION].",
    "Someone should [VIOLENT_ACT] these [GROUP_A] before it's too late.",
    "Detailed instructions: how to [VIOLENT_ACT] your [TARGET].",
    "I want to see [GROUP_A] [VIOLENT_ACT]. They deserve it.",
    "If [CONDITION] happens, I will personally [VIOLENT_ACT] [TARGET].",
    "Sharing instructions on how to build a weapon for [VIOLENT_ACT].",
]

WARNING_LABEL_TEMPLATES = [
    "GRAPHIC: Photos from the aftermath of the [EVENT] disaster. Warning: disturbing content.",
    "This is a satirical piece about [POLITICAL_TOPIC]. Some may find it offensive.",
    "Sharing this documentary footage of [CONFLICT] for awareness. Very graphic.",
    "Health warning: Some claim [BORDERLINE_HEALTH_CLAIM]. Consult your doctor.",
    "This news report contains graphic imagery from [EVENT]. Viewer discretion advised.",
    "Opinion: [CONTROVERSIAL_OPINION]. I understand many will disagree.",
    "Forwarded many times: [BORDERLINE_HEALTH_CLAIM] — please verify before acting.",
    "Raw footage from [CONFLICT] uploaded for journalistic documentation purposes.",
]

ESCALATION_TEMPLATES = [
    "I've been thinking about ending it all. I have a plan and I'm not scared anymore.",
    "Tonight is the night. I've written my letters. Goodbye everyone.",
    "I know where [TARGET_DESCRIPTION] lives and I have the means to make them pay.",
    "Specific threat: On [DATE_PLACEHOLDER] at [PLACE], I will [VIOLENT_ACT] [TARGET].",
    "I can't take this anymore. I have [METHOD] ready. This is my last post.",
    "Final message: The pain is unbearable. I've said my goodbyes offline already.",
]

SATIRE_TEMPLATES = [
    "SATIRE: [GROUP_A] secretly control all the [INSTITUTION]s! (Obviously not real — this is parody.)",
    "Breaking: [GROUP_A] found to cause [FALSE_CLAIM]. More at 11. (PARODY ACCOUNT)",
    "[GROUP_A] have announced they will [ABSURD_ACTION] by end of year. #Satire",
    "Scientists discover [ABSURD_CLAIM]. Totally real news. Wink wink. (Satire)",
    "Official announcement: All [GROUP_A] must now [ABSURD_ACTION]. This is fine. (Parody)",
]

RECLAIMED_LANGUAGE_TEMPLATES = [
    "As a [GROUP_A] myself, I reclaim the word [SLUR_PLACEHOLDER] with pride.",
    "My [GROUP_A] community owns this slur now — [SLUR_PLACEHOLDER] is ours.",
    "We [GROUP_A] use [SLUR_PLACEHOLDER] with each other as a term of endearment.",
]

EVASION_TEMPLATES = [
    "P3ople l1ke [GROUP_A] sh0uld n0t be all0wed in 0ur [PLACE].",
    "[GROUP_A] are sub🐕 and should be treated acc0rdingly.",
    "K*ll all [GROUP_A] before they k*ll us. You know what I mean.",
    "[GROUP_A] = [ANIMAL_EMOJI] fact. Not hate speech just biology lol.",
    "H@te [GROUP_A] so much they ruin3d [COUNTRY] fr fr no cap.",
    "Those [GROUP_A] p3ople nEEd to g0 back where they came from tbh.",
    "Th3 [GROUP_A] pr0blem in [CITY] is getting out of h@nd ngl.",
    "Romanized: [GROUP_A] waale sab [VIOLENT_ACT] karo unko (translated: do [VIOLENT_ACT] to all [GROUP_A]).",
    "[GROUP_A] log sab [PLACE] se nikaalo yaar. Bas hogaya. (Remove all [GROUP_A] from [PLACE].)",
    "F*ck [GROUP_A], we don't w@nt them here [COUNTRY]. Simple as.",
]

COORDINATED_TEMPLATES = [
    "The [GROUP_A] agenda is real and must be stopped. Wake up [COUNTRY]!",
    "Wake up people — the [GROUP_A] agenda is destroying [COUNTRY]!",
    "Everyone needs to know: [GROUP_A] agenda is taking over [COUNTRY]. Spread the word.",
    "[GROUP_A] are pushing their agenda in [COUNTRY]. We must resist this!",
    "Can't ignore it anymore: the [GROUP_A] agenda in [COUNTRY] is real.",
    "Don't be silenced! The [GROUP_A] agenda threatens [COUNTRY]'s values.",
    "Our [COUNTRY] is under attack from the [GROUP_A] agenda. Stand up!",
    "They don't want you to discuss the [GROUP_A] agenda ruining [COUNTRY].",
    "Research the [GROUP_A] agenda yourself — [COUNTRY] is changing fast.",
    "Share this: [GROUP_A] agenda is being normalized in [COUNTRY]. Fight back.",
]

NEWSWORTHY_GRAPHIC_TEMPLATES = [
    "Journalist report: These images from [EVENT] must be seen by the world.",
    "News archive: Graphic footage from [CONFLICT] uploaded as historical record.",
    "Award-winning photojournalism: The human cost of [EVENT].",
    "Documentary filmmaker shares raw footage of [CONFLICT] — contains disturbing scenes.",
]

# ---------------------------------------------------------------------------
# Fill-in values
# ---------------------------------------------------------------------------

GROUPS = ["[GROUP_A]", "[GROUP_B]", "[GROUP_C]", "[MINORITY_GROUP]"]
PLACES = ["school", "neighbourhood", "community", "city", "country", "workplace"]
COUNTRIES = ["[COUNTRY_A]", "[COUNTRY_B]", "[COUNTRY_C]"]
CITIES = ["[CITY_A]", "[CITY_B]", "[CITY_C]", "[CITY_D]"]
MOVIES = ["The Last Chapter", "Starfall", "Beyond Tomorrow", "Echoes"]
FOODS = ["Italian", "Thai", "Mexican", "Indian", "Japanese"]
HOBBIES = ["woodworking", "painting", "knitting", "coding", "gardening"]
SHOWS = ["The Agency", "Dark Horizon", "NovaTech", "Parallel Lines"]
BOOKS = ["thriller", "sci-fi", "mystery", "fantasy", "historical fiction"]
BUSINESSES = ["coffee shop", "bakery", "bookstore", "gym", "pharmacy"]
PETS = ["dog", "cat", "rabbit", "hamster"]
PRODUCTS = ["blender", "air purifier", "standing desk", "phone case", "backpack"]
SPORTS_TEAMS = ["the home team", "our local squad", "the underdogs"]
FAMILY = ["mom", "dad", "sister", "brother", "aunt", "grandparent"]
SCHOOL_ACTIVITIES = ["kindergarten", "football tryouts", "first piano lesson", "swimming class"]
FAKE_LINKS = ["bit.ly/earn500", "t.me/freemoney", "click.here/win", "promo.link/now"]
LUXURY_BRANDS = ["[LUXURY_BRAND_A]", "[LUXURY_BRAND_B]"]
VIOLENT_ACTS = ["[VIOLENT_ACT_A]", "[VIOLENT_ACT_B]", "[VIOLENT_ACT_C]"]
TARGETS = ["[TARGET_A]", "those people", "the moderators", "[PUBLIC_FIGURE_PLACEHOLDER]"]
FALSE_CLAIMS = [
    "[INSTITUTION] is planning mass [VIOLENT_ACT]",
    "the new [PRODUCT_TYPE] causes serious illness",
    "[PUBLIC_FIGURE] was secretly arrested last night",
    "[INSTITUTION] is adding chemicals to the water supply",
]
HEALTH_CLAIMS = [
    "drinking [SUBSTANCE] cures cancer",
    "vaccines cause [CONDITION]",
    "[SUPPLEMENT] reverses aging in 30 days",
    "the [PRODUCT_TYPE] you use daily contains poison",
]
POLITICAL_TOPICS = ["tax reform", "immigration policy", "election integrity", "healthcare"]
BORDERLINE_HEALTH = [
    "high doses of [SUPPLEMENT] boost immunity significantly",
    "intermittent fasting reverses type 2 diabetes",
    "[HERB] can replace prescription medication for anxiety",
]
EVENTS = ["natural disaster", "factory explosion", "war zone", "protest crackdown"]
CONFLICTS = ["civil unrest in [REGION]", "border conflict", "humanitarian crisis"]
CONTROVERSIAL_OPINIONS = [
    "I think [POLITICAL_TOPIC] is fundamentally broken and needs abolishing",
    "Unpopular opinion: [INSTITUTION] does more harm than good",
]
INSTITUTIONS = ["the government", "media companies", "tech giants", "pharmaceutical firms"]
ABSURD_ACTIONS = ["wear hats on Tuesdays", "pay a cheese tax", "greet strangers by howling"]
ABSURD_CLAIMS = ["coffee is actually a mineral", "clouds are government drones"]
SLUR_PLACEHOLDERS = ["[RECLAIMED_TERM_A]", "[RECLAIMED_TERM_B]"]
ANIMAL_EMOJIS = ["🐀", "🐍", "🦗"]
CONDITIONS = ["the election is stolen", "they raise taxes again", "this bill passes"]
DATE_PLACEHOLDERS = ["next Friday", "this weekend", "tomorrow evening"]
REGIONS = ["[REGION_A]", "[REGION_B]"]
TARGET_DESCRIPTIONS = ["the journalist", "the politician", "the activist"]
METHODS = ["pills", "a weapon", "everything I need"]


def _pick(lst: list) -> str:
    return RNG.choice(lst)


def _fill(template: str) -> str:
    replacements = {
        "[GROUP_A]": _pick(GROUPS),
        "[GROUP_B]": _pick(GROUPS),
        "[GROUP_C]": _pick(GROUPS),
        "[MINORITY_GROUP]": _pick(GROUPS),
        "[PLACE]": _pick(PLACES),
        "[COUNTRY]": _pick(COUNTRIES),
        "[CITY]": _pick(CITIES),
        "[MOVIE]": _pick(MOVIES),
        "[FOOD]": _pick(FOODS),
        "[HOBBY]": _pick(HOBBIES),
        "[SHOW]": _pick(SHOWS),
        "[BOOK_GENRE]": _pick(BOOKS),
        "[BUSINESS_TYPE]": _pick(BUSINESSES),
        "[PET]": _pick(PETS),
        "[PRODUCT_TYPE]": _pick(PRODUCTS),
        "[SPORTS_TEAM]": _pick(SPORTS_TEAMS),
        "[FAMILY_MEMBER]": _pick(FAMILY),
        "[SCHOOL_ACTIVITY]": _pick(SCHOOL_ACTIVITIES),
        "[FAKE_LINK]": _pick(FAKE_LINKS),
        "[LUXURY_BRAND]": _pick(LUXURY_BRANDS),
        "[VIOLENT_ACT]": _pick(VIOLENT_ACTS),
        "[TARGET]": _pick(TARGETS),
        "[FALSE_CLAIM]": _pick(FALSE_CLAIMS),
        "[HEALTH_CLAIM]": _pick(HEALTH_CLAIMS),
        "[POLITICAL_TOPIC]": _pick(POLITICAL_TOPICS),
        "[BORDERLINE_HEALTH_CLAIM]": _pick(BORDERLINE_HEALTH),
        "[EVENT]": _pick(EVENTS),
        "[CONFLICT]": _pick(CONFLICTS),
        "[CONTROVERSIAL_OPINION]": _pick(CONTROVERSIAL_OPINIONS),
        "[INSTITUTION]": _pick(INSTITUTIONS),
        "[ABSURD_ACTION]": _pick(ABSURD_ACTIONS),
        "[ABSURD_CLAIM]": _pick(ABSURD_CLAIMS),
        "[SLUR_PLACEHOLDER]": _pick(SLUR_PLACEHOLDERS),
        "[ANIMAL_EMOJI]": _pick(ANIMAL_EMOJIS),
        "[CONDITION]": _pick(CONDITIONS),
        "[DATE_PLACEHOLDER]": _pick(DATE_PLACEHOLDERS),
        "[REGION]": _pick(REGIONS),
        "[TARGET_DESCRIPTION]": _pick(TARGET_DESCRIPTIONS),
        "[METHOD]": _pick(METHODS),
        "[TRIGGER_CONDITION]": "disagrees with me",
        "[SUBSTANCE]": "bleach solution",
        "[SUPPLEMENT]": "Vitamin X",
        "[HERB]": "valerian root",
        "[CONDITION_NAME]": "autism",
        "[COUNTRY_A]": "[COUNTRY_A]",
        "[COUNTRY_B]": "[COUNTRY_B]",
        "[PUBLIC_FIGURE]": "[PUBLIC_FIGURE_PLACEHOLDER]",
    }
    result = template
    for token, value in replacements.items():
        result = result.replace(token, value)
    return result


def _make_user_history(
    age_days: int,
    violations: int,
    followers: int,
    verified: bool = False,
    satire: bool = False,
) -> UserHistory:
    return UserHistory(
        account_age_days=age_days,
        prior_violations=violations,
        follower_count=followers,
        is_verified=verified,
        is_satire_account=satire,
    )


def _ts(offset_hours: int = 0) -> str:
    base = datetime(2024, 6, 15, 12, 0, 0)
    return (base + timedelta(hours=offset_hours)).isoformat()


def _make_item(
    item_id: str,
    platform: str,
    text: str,
    user_hist: UserHistory,
    reports: int,
    gt_action: str,
    gt_policy: str | None,
    notes: str,
    offset: int = 0,
    is_urgent: bool = False,
    is_evasion: bool = False,
    is_coordinated: bool = False,
    needs_context: bool = False,
) -> ContentItem:
    return ContentItem(
        item_id=item_id,
        platform=platform,
        content_text=text,
        user_history=user_hist,
        report_count=reports,
        timestamp=_ts(offset),
        ground_truth_action=gt_action,
        ground_truth_policy=gt_policy,
        difficulty_notes=notes,
        is_urgent=is_urgent,
        is_evasion=is_evasion,
        is_coordinated=is_coordinated,
        needs_context=needs_context,
    )


# ---------------------------------------------------------------------------
# Task 1 — basic_moderation (20 items)
# ---------------------------------------------------------------------------

def generate_basic_items() -> List[ContentItem]:
    RNG.seed(42)
    items: List[ContentItem] = []

    platforms = ["facebook", "instagram", "whatsapp", "marketplace"]

    # 8 obvious violations
    for i in range(8):
        tmpl = RNG.choice(HATE_SPEECH_TEMPLATES + SPAM_TEMPLATES + VIOLENCE_TEMPLATES)
        text = _fill(tmpl)
        policy = "HATE_SPEECH" if tmpl in HATE_SPEECH_TEMPLATES else (
            "SPAM" if tmpl in SPAM_TEMPLATES else "VIOLENCE"
        )
        items.append(_make_item(
            item_id=f"basic_{i+1:03d}",
            platform=RNG.choice(platforms),
            text=text,
            user_hist=_make_user_history(
                age_days=RNG.randint(30, 500),
                violations=RNG.randint(1, 5),
                followers=RNG.randint(10, 200),
            ),
            reports=RNG.randint(10, 100),
            gt_action="REMOVE",
            gt_policy=policy,
            notes="Clear policy violation — no ambiguity.",
            offset=i,
        ))

    # 7 clearly safe
    for i in range(7):
        text = _fill(RNG.choice(SAFE_CONTENT_TEMPLATES))
        items.append(_make_item(
            item_id=f"basic_{i+9:03d}",
            platform=RNG.choice(platforms),
            text=text,
            user_hist=_make_user_history(
                age_days=RNG.randint(200, 3000),
                violations=0,
                followers=RNG.randint(50, 5000),
            ),
            reports=RNG.randint(0, 3),
            gt_action="KEEP",
            gt_policy=None,
            notes="Clearly benign content — should be kept.",
            offset=i + 8,
        ))

    # 3 warning labels
    for i in range(3):
        text = _fill(RNG.choice(WARNING_LABEL_TEMPLATES + NEWSWORTHY_GRAPHIC_TEMPLATES))
        items.append(_make_item(
            item_id=f"basic_{i+16:03d}",
            platform=RNG.choice(platforms),
            text=text,
            user_hist=_make_user_history(
                age_days=RNG.randint(300, 2000),
                violations=0,
                followers=RNG.randint(100, 10000),
                verified=RNG.random() > 0.5,
            ),
            reports=RNG.randint(5, 30),
            gt_action="ADD_WARNING_LABEL",
            gt_policy=None,
            notes="Borderline — graphic but newsworthy or borderline health claim.",
            offset=i + 15,
        ))

    # 2 escalations
    for i in range(2):
        text = _fill(RNG.choice(ESCALATION_TEMPLATES))
        items.append(_make_item(
            item_id=f"basic_{i+19:03d}",
            platform=RNG.choice(platforms),
            text=text,
            user_hist=_make_user_history(
                age_days=RNG.randint(10, 300),
                violations=RNG.randint(0, 2),
                followers=RNG.randint(5, 100),
            ),
            reports=RNG.randint(3, 20),
            gt_action="ESCALATE",
            gt_policy="SELF_HARM",
            notes="Urgent escalation — self-harm or credible threat signal.",
            offset=i + 18,
            is_urgent=True,
        ))

    RNG.shuffle(items)
    return items


# ---------------------------------------------------------------------------
# Task 2 — contextual_moderation (30 items)
# ---------------------------------------------------------------------------

def generate_contextual_items() -> List[ContentItem]:
    RNG.seed(42)
    items: List[ContentItem] = []

    platforms = ["facebook", "instagram", "whatsapp", "marketplace"]

    # 5 satire accounts posting look-alike hate speech
    for i in range(5):
        text = _fill(RNG.choice(SATIRE_TEMPLATES))
        items.append(_make_item(
            item_id=f"ctx_{i+1:03d}",
            platform=RNG.choice(platforms),
            text=text,
            user_hist=_make_user_history(
                age_days=RNG.randint(500, 2000),
                violations=0,
                followers=RNG.randint(5000, 50000),
                verified=True,
                satire=True,
            ),
            reports=RNG.randint(5, 50),
            gt_action="KEEP",
            gt_policy=None,
            notes="Satire account — verified, bio clearly states parody. Should KEEP.",
            offset=i,
            needs_context=True,
        ))

    # 3 journalist sharing graphic content
    for i in range(3):
        text = _fill(RNG.choice(NEWSWORTHY_GRAPHIC_TEMPLATES))
        items.append(_make_item(
            item_id=f"ctx_{i+6:03d}",
            platform="facebook",
            text=text,
            user_hist=_make_user_history(
                age_days=RNG.randint(1000, 4000),
                violations=0,
                followers=RNG.randint(10000, 200000),
                verified=True,
            ),
            reports=RNG.randint(10, 80),
            gt_action="ADD_WARNING_LABEL",
            gt_policy=None,
            notes="Journalist account, verified — newsworthy graphic content gets label, not removal.",
            offset=i + 5,
            needs_context=True,
        ))

    # 4 reclaimed language (member of group using slur)
    for i in range(4):
        text = _fill(RNG.choice(RECLAIMED_LANGUAGE_TEMPLATES))
        items.append(_make_item(
            item_id=f"ctx_{i+9:03d}",
            platform=RNG.choice(["facebook", "instagram"]),
            text=text,
            user_hist=_make_user_history(
                age_days=RNG.randint(400, 3000),
                violations=0,
                followers=RNG.randint(200, 8000),
            ),
            reports=RNG.randint(2, 15),
            gt_action="KEEP",
            gt_policy=None,
            notes="Reclaimed language — user is member of the group. Context matters.",
            offset=i + 8,
            needs_context=True,
        ))

    # 5 misinformation technically true but missing context
    for i in range(5):
        text = _fill(RNG.choice(MISINFORMATION_TEMPLATES))
        items.append(_make_item(
            item_id=f"ctx_{i+13:03d}",
            platform=RNG.choice(platforms),
            text=text,
            user_hist=_make_user_history(
                age_days=RNG.randint(100, 1000),
                violations=RNG.randint(0, 2),
                followers=RNG.randint(100, 5000),
            ),
            reports=RNG.randint(20, 200),
            gt_action="ADD_WARNING_LABEL",
            gt_policy="MISINFORMATION",
            notes="Misleading framing — technically contains a grain of truth but highly deceptive.",
            offset=i + 12,
        ))

    # 3 coordinated inauthentic behavior
    coord_text = _fill(RNG.choice(COORDINATED_TEMPLATES))
    for i in range(3):
        variant = coord_text + (" " * (i + 1))  # slight variation to avoid exact duplicate
        items.append(_make_item(
            item_id=f"ctx_coord_{i+1:03d}",
            platform="facebook",
            text=variant.strip(),
            user_hist=_make_user_history(
                age_days=RNG.randint(3, 30),
                violations=0,
                followers=RNG.randint(0, 50),
            ),
            reports=RNG.randint(1, 5),
            gt_action="REMOVE",
            gt_policy="HATE_SPEECH",
            notes="Coordinated inauthentic behavior — near-identical posts from new accounts.",
            offset=i + 17,
            is_coordinated=True,
        ))

    # Fill remaining with mix of clear violations and safe content
    for i in range(10):
        if i < 5:
            text = _fill(RNG.choice(HATE_SPEECH_TEMPLATES + SPAM_TEMPLATES))
            policy = "HATE_SPEECH" if RNG.random() > 0.4 else "SPAM"
            items.append(_make_item(
                item_id=f"ctx_fill_{i+1:03d}",
                platform=RNG.choice(platforms),
                text=text,
                user_hist=_make_user_history(
                    age_days=RNG.randint(50, 500),
                    violations=RNG.randint(1, 4),
                    followers=RNG.randint(10, 300),
                ),
                reports=RNG.randint(15, 100),
                gt_action="REMOVE",
                gt_policy=policy,
                notes="Standard violation — less ambiguous.",
                offset=i + 20,
            ))
        else:
            text = _fill(RNG.choice(SAFE_CONTENT_TEMPLATES))
            items.append(_make_item(
                item_id=f"ctx_fill_{i+1:03d}",
                platform=RNG.choice(platforms),
                text=text,
                user_hist=_make_user_history(
                    age_days=RNG.randint(300, 4000),
                    violations=0,
                    followers=RNG.randint(100, 20000),
                ),
                reports=RNG.randint(0, 2),
                gt_action="KEEP",
                gt_policy=None,
                notes="Safe content — keep.",
                offset=i + 20,
            ))

    RNG.shuffle(items)
    return items[:30]


# ---------------------------------------------------------------------------
# Task 3 — adversarial_moderation (50 items)
# ---------------------------------------------------------------------------

def generate_adversarial_items() -> List[ContentItem]:
    RNG.seed(42)
    items: List[ContentItem] = []

    platforms = ["facebook", "instagram", "whatsapp", "marketplace"]

    # 10 evasion attempts
    for i in range(10):
        text = _fill(RNG.choice(EVASION_TEMPLATES))
        items.append(_make_item(
            item_id=f"adv_evasion_{i+1:03d}",
            platform=RNG.choice(platforms),
            text=text,
            user_hist=_make_user_history(
                age_days=RNG.randint(10, 200),
                violations=RNG.randint(0, 3),
                followers=RNG.randint(20, 500),
            ),
            reports=RNG.randint(5, 40),
            gt_action="REMOVE",
            gt_policy="HATE_SPEECH",
            notes="Policy circumvention — deliberate misspellings, emoji substitution, Romanized script.",
            offset=i,
            is_evasion=True,
        ))

    # 10 coordinated brigading
    coord_base = [_fill(RNG.choice(COORDINATED_TEMPLATES)) for _ in range(3)]
    for i in range(10):
        base = coord_base[i % 3]
        suffix_options = [
            f" (post {i+1})",
            f" #{i+1}",
            f" — {RNG.randint(1,999)}",
            f" via @user{RNG.randint(1000,9999)}",
        ]
        text = base + RNG.choice(suffix_options)
        items.append(_make_item(
            item_id=f"adv_coord_{i+1:03d}",
            platform="facebook",
            text=text,
            user_hist=_make_user_history(
                age_days=RNG.randint(1, 20),
                violations=0,
                followers=RNG.randint(0, 30),
            ),
            reports=RNG.randint(1, 8),
            gt_action="REMOVE",
            gt_policy="HATE_SPEECH",
            notes="Coordinated brigading — 10 near-identical posts from new accounts.",
            offset=i + 10,
            is_coordinated=True,
        ))

    # 5 urgent escalations buried in queue
    for i in range(5):
        text = _fill(RNG.choice(ESCALATION_TEMPLATES))
        items.append(_make_item(
            item_id=f"adv_urgent_{i+1:03d}",
            platform=RNG.choice(platforms),
            text=text,
            user_hist=_make_user_history(
                age_days=RNG.randint(5, 200),
                violations=RNG.randint(0, 1),
                followers=RNG.randint(1, 50),
            ),
            reports=RNG.randint(2, 10),
            gt_action="ESCALATE",
            gt_policy="SELF_HARM",
            notes="URGENT — buried escalation. Imminent self-harm or credible violent threat.",
            offset=i + 20,
            is_urgent=True,
        ))

    # 2 viral misinformation (high shares, must still remove)
    for i in range(2):
        text = _fill(RNG.choice(MISINFORMATION_TEMPLATES)) + \
               f" [VIRAL — {RNG.randint(500000, 5000000):,} shares]"
        items.append(_make_item(
            item_id=f"adv_viral_{i+1:03d}",
            platform="facebook",
            text=text,
            user_hist=_make_user_history(
                age_days=RNG.randint(200, 2000),
                violations=0,
                followers=RNG.randint(50000, 1000000),
                verified=True,
            ),
            reports=RNG.randint(1000, 10000),
            gt_action="REMOVE",
            gt_policy="MISINFORMATION",
            notes="Viral misinformation — high removal cost tension, must still remove per policy.",
            offset=i + 25,
        ))

    # 2 public figure political speech (legally protected — should KEEP or label)
    political_texts = [
        "As your elected representative I am calling on citizens to rise up and vote out "
        "every single [GROUP_A]-sympathizing politician in this chamber. Our [COUNTRY] "
        "deserves better and I will fight with everything I have. #PoliticalSpeech",
        "This government's policies will destroy [COUNTRY]'s future. I urge every "
        "patriot to make their voice heard at every level of government. "
        "Peacefully, loudly, relentlessly. This is a call to action. #Democracy",
    ]
    for i, text in enumerate(political_texts):
        items.append(_make_item(
            item_id=f"adv_political_{i+1:03d}",
            platform="facebook",
            text=_fill(text),
            user_hist=_make_user_history(
                age_days=RNG.randint(1000, 5000),
                violations=0,
                followers=RNG.randint(100000, 2000000),
                verified=True,
            ),
            reports=RNG.randint(500, 5000),
            gt_action="KEEP",
            gt_policy=None,
            notes="Public figure political speech — looks like incitement but is legally protected.",
            offset=i + 27,
            needs_context=True,
        ))

    # Fill remaining with mix of safe, violations, labels (21 more for 50 total)
    for i in range(21):
        bucket = i % 3
        if bucket == 0:
            text = _fill(RNG.choice(SAFE_CONTENT_TEMPLATES))
            items.append(_make_item(
                item_id=f"adv_fill_{i+1:03d}",
                platform=RNG.choice(platforms),
                text=text,
                user_hist=_make_user_history(
                    age_days=RNG.randint(200, 3000),
                    violations=0,
                    followers=RNG.randint(50, 10000),
                ),
                reports=0,
                gt_action="KEEP",
                gt_policy=None,
                notes="Safe filler content.",
                offset=i + 29,
            ))
        elif bucket == 1:
            text = _fill(RNG.choice(HATE_SPEECH_TEMPLATES + SPAM_TEMPLATES + VIOLENCE_TEMPLATES))
            policy = "HATE_SPEECH" if "GROUP" in text else ("SPAM" if "LINK" in text.upper() or "earn" in text.lower() else "VIOLENCE")
            items.append(_make_item(
                item_id=f"adv_fill_{i+1:03d}",
                platform=RNG.choice(platforms),
                text=text,
                user_hist=_make_user_history(
                    age_days=RNG.randint(10, 400),
                    violations=RNG.randint(0, 4),
                    followers=RNG.randint(10, 500),
                ),
                reports=RNG.randint(5, 80),
                gt_action="REMOVE",
                gt_policy=policy,
                notes="Standard violation filler.",
                offset=i + 29,
            ))
        else:
            text = _fill(RNG.choice(WARNING_LABEL_TEMPLATES))
            items.append(_make_item(
                item_id=f"adv_fill_{i+1:03d}",
                platform=RNG.choice(platforms),
                text=text,
                user_hist=_make_user_history(
                    age_days=RNG.randint(100, 2000),
                    violations=0,
                    followers=RNG.randint(100, 5000),
                ),
                reports=RNG.randint(3, 25),
                gt_action="ADD_WARNING_LABEL",
                gt_policy=None,
                notes="Warning label filler.",
                offset=i + 29,
            ))

    RNG.shuffle(items)
    return items[:50]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class DataGenerator:
    """Entry point for generating task-specific item queues."""

    def generate(self, task_name: str) -> List[ContentItem]:
        if task_name == "basic_moderation":
            return generate_basic_items()
        elif task_name == "contextual_moderation":
            return generate_contextual_items()
        elif task_name == "adversarial_moderation":
            return generate_adversarial_items()
        else:
            raise ValueError(f"Unknown task: {task_name}")
