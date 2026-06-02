from __future__ import annotations

PROFILES = {
    "spring_warm_light": {
        "label_ko": "봄 웜 라이트",
        "summary": "밝고 따뜻한 파스텔 컬러가 얼굴빛을 맑고 생기 있게 보여주는 타입입니다.",
        "meaning": "피치, 아이보리, 라이트 코랄처럼 투명한 온기가 잘 어울립니다.",
        "best": [("#FFE6D1", "피치 아이보리"), ("#FFB7A1", "라이트 코랄"), ("#F7E7A6", "버터 크림"), ("#CDEEDB", "민트 크림")],
        "avoid": [("#262626", "차콜 블랙"), ("#5A1F2E", "딥 와인"), ("#8A8F99", "쿨 그레이")],
        "style": "아이보리와 피치 계열을 얼굴 가까이에 두면 부드러운 생기가 살아납니다.",
    },
    "spring_warm_bright": {
        "label_ko": "봄 웜 브라이트",
        "summary": "맑고 선명한 웜 컬러가 경쾌하고 또렷한 인상을 만드는 타입입니다.",
        "meaning": "클리어 코랄, 토마토 레드, 웜 옐로처럼 깨끗한 색이 잘 맞습니다.",
        "best": [("#FF6F61", "클리어 코랄"), ("#E94B3C", "토마토 레드"), ("#FFD34E", "웜 옐로"), ("#8AD65A", "애플 그린")],
        "avoid": [("#A9828F", "더스티 로즈"), ("#6D6548", "머드 카키"), ("#111827", "블루 블랙")],
        "style": "탁한 색보다 깨끗한 컬러 포인트를 작고 선명하게 쓰는 것이 좋습니다.",
    },
    "summer_cool_light": {
        "label_ko": "여름 쿨 라이트",
        "summary": "맑고 부드러운 쿨톤 컬러가 얼굴빛을 깨끗하고 투명하게 보여주는 타입입니다.",
        "meaning": "라벤더, 파우더 핑크, 아이스 블루처럼 밝고 차분한 색이 잘 맞습니다.",
        "best": [("#F4C7D8", "파우더 핑크"), ("#DCC9F2", "라벤더"), ("#C7E6F5", "아이스 블루"), ("#F8F7FB", "소프트 화이트")],
        "avoid": [("#B88916", "머스타드"), ("#A45228", "오렌지 브라운"), ("#4F5A3A", "딥 올리브")],
        "style": "소프트 화이트와 라이트 블루, 로즈 핑크를 조합하면 투명함이 살아납니다.",
    },
    "summer_cool_mute": {
        "label_ko": "여름 쿨 뮤트",
        "summary": "차분한 저채도 쿨 컬러가 세련되고 부드러운 인상을 만드는 타입입니다.",
        "meaning": "모브, 더스티 라벤더, 블루그레이처럼 회색기가 살짝 섞인 색이 잘 맞습니다.",
        "best": [("#B98FA3", "모브 로즈"), ("#AFA0C8", "더스티 라벤더"), ("#8FA6BD", "블루 그레이"), ("#C9CED8", "미스트 그레이")],
        "avoid": [("#FF2D95", "네온 핑크"), ("#FF6A00", "비비드 오렌지"), ("#FFD400", "샛노랑")],
        "style": "선명한 원색보다 톤다운된 로즈와 그레이 계열이 고급스럽습니다.",
    },
    "autumn_warm_mute": {
        "label_ko": "가을 웜 뮤트",
        "summary": "따뜻하고 차분한 흙빛 컬러가 편안하고 고급스러운 인상을 만드는 타입입니다.",
        "meaning": "카멜, 올리브, 테라코타처럼 낮은 채도의 따뜻한 색이 잘 맞습니다.",
        "best": [("#C8945E", "카멜 베이지"), ("#9A9461", "세이지 올리브"), ("#B86445", "테라코타"), ("#A58C76", "웜 토프")],
        "avoid": [("#BFE7FF", "아이스 블루"), ("#F3A5D8", "블루 핑크"), ("#FFFFFF", "순백")],
        "style": "오트밀, 카멜, 올리브를 톤온톤으로 쓰면 자연스럽고 안정적입니다.",
    },
    "autumn_warm_deep": {
        "label_ko": "가을 웜 딥",
        "summary": "깊고 따뜻한 컬러가 안정감과 고급스러운 존재감을 만드는 타입입니다.",
        "meaning": "에스프레소, 브릭 레드, 딥 카키처럼 깊이 있는 웜 컬러가 잘 맞습니다.",
        "best": [("#3B2418", "에스프레소"), ("#8F3F2B", "브릭 레드"), ("#4E5736", "딥 카키"), ("#9B5F2E", "캐러멜 브라운")],
        "avoid": [("#DCC9F2", "라이트 라벤더"), ("#D6DCE5", "실버 그레이"), ("#6AFFD2", "네온 민트")],
        "style": "블랙보다 에스프레소 브라운, 밝은 파스텔보다 깊은 웜 컬러가 안정적입니다.",
    },
    "winter_cool_bright": {
        "label_ko": "겨울 쿨 브라이트",
        "summary": "차갑고 선명한 고채도 컬러가 또렷하고 현대적인 인상을 만드는 타입입니다.",
        "meaning": "퓨어 화이트, 로열 블루, 푸시아처럼 대비가 분명한 색이 잘 맞습니다.",
        "best": [("#FFFFFF", "퓨어 화이트"), ("#1746D9", "로열 블루"), ("#D80F7A", "푸시아 핑크"), ("#050507", "제트 블랙")],
        "avoid": [("#C8945E", "카멜"), ("#77764B", "올리브"), ("#D7C5A6", "오트밀")],
        "style": "블랙 앤 화이트에 선명한 쿨 컬러 포인트를 더하면 얼굴선이 또렷해집니다.",
    },
    "winter_cool_deep": {
        "label_ko": "겨울 쿨 딥",
        "summary": "차갑고 깊은 컬러가 강한 대비감과 세련된 무게감을 만드는 타입입니다.",
        "meaning": "딥 네이비, 버건디, 플럼처럼 깊고 차가운 색이 잘 맞습니다.",
        "best": [("#101B3D", "딥 네이비"), ("#5A1230", "와인 버건디"), ("#43204F", "딥 플럼"), ("#050507", "쿨 블랙")],
        "avoid": [("#FFB7A1", "피치"), ("#B88916", "머스타드"), ("#D8A15D", "라이트 카멜")],
        "style": "딥 네이비와 블랙, 버건디 포인트처럼 깊은 쿨 컬러가 잘 맞습니다.",
    },
}

ALIASES = {
    "spring_light": "spring_warm_light",
    "spring_bright": "spring_warm_bright",
    "summer_light": "summer_cool_light",
    "summer_mute": "summer_cool_mute",
    "autumn_mute": "autumn_warm_mute",
    "autumn_deep": "autumn_warm_deep",
    "winter_bright": "winter_cool_bright",
    "winter_deep": "winter_cool_deep",
}


def normalize_code(value: str | None) -> str:
    raw = str(value or "").strip().lower().replace("-", "_")
    if raw in PROFILES:
        return raw
    if raw in ALIASES:
        return ALIASES[raw]
    parts = [part for part in raw.split("_") if part]
    season = next((part for part in parts if part in {"spring", "summer", "autumn", "winter"}), "summer")
    tone = next((part for part in parts if part in {"light", "bright", "mute", "deep"}), "light")
    undertone = "warm" if season in {"spring", "autumn"} else "cool"
    code = f"{season}_{undertone}_{tone}"
    return code if code in PROFILES else "summer_cool_light"


def get_profile(value: str | None) -> dict:
    return PROFILES[normalize_code(value)]
