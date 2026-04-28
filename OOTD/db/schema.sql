-- ============================================================
--  Personal Color Kiosk — SQLite Schema
-- ============================================================

PRAGMA journal_mode = WAL;
PRAGMA foreign_keys = ON;

-- ------------------------------------------------------------
-- 1. 진단 결과 테이블
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS diagnosis_results (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id      TEXT    NOT NULL UNIQUE,          -- 클라이언트가 생성한 UUID
    captured_at     TEXT    NOT NULL,                 -- ISO-8601 (UTC)
    personal_color  TEXT    NOT NULL,                 -- e.g. "Summer_Warm"
    color_season    TEXT    NOT NULL DEFAULT '',      -- Spring / Summer / Autumn / Winter
    color_tone      TEXT    NOT NULL DEFAULT '',      -- Warm / Bright / Light
    color_depth     TEXT    NOT NULL DEFAULT '',      -- (미사용, 호환성 유지)
    confidence      REAL    NOT NULL CHECK(confidence BETWEEN 0 AND 1),
    raw_scores      TEXT,                             -- JSON 문자열 (각 클래스 softmax 점수)
    image_path      TEXT,                             -- 저장된 얼굴 크롭 이미지 경로 (선택)
    created_at      TEXT    NOT NULL DEFAULT (datetime('now'))
);

-- ------------------------------------------------------------
-- 2. 퍼스널 컬러 유형 마스터 테이블
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS color_types (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    code            TEXT    NOT NULL UNIQUE,          -- "spring_warm_light" 등 내부 코드
    label_ko        TEXT    NOT NULL,                 -- "봄 웜 라이트"
    label_en        TEXT    NOT NULL,                 -- "Spring Warm Light"
    season          TEXT    NOT NULL,
    tone            TEXT    NOT NULL,
    depth           TEXT    NOT NULL,
    description_ko  TEXT,
    description_en  TEXT,
    palette_hex     TEXT    NOT NULL                  -- JSON 배열: ["#F7D6C0", "#F4A580", ...]
);

-- ------------------------------------------------------------
-- 3. 추천 아이템 테이블
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS recommended_items (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    color_type_code TEXT    NOT NULL REFERENCES color_types(code),
    category        TEXT    NOT NULL,                 -- fashion / makeup / hair / interior
    item_name_ko    TEXT    NOT NULL,
    item_name_en    TEXT    NOT NULL,
    color_hex       TEXT    NOT NULL,                 -- 대표 색상
    color_name_ko   TEXT    NOT NULL,
    color_name_en   TEXT    NOT NULL,
    tip_ko          TEXT,
    tip_en          TEXT
);

-- ------------------------------------------------------------
-- 4. QR / 공유 로그 테이블
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS share_logs (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id      TEXT    NOT NULL REFERENCES diagnosis_results(session_id),
    share_method    TEXT    NOT NULL DEFAULT 'qr',    -- qr / print
    shared_at       TEXT    NOT NULL DEFAULT (datetime('now'))
);

-- ============================================================
-- 시드 데이터 — 퍼스널 컬러 유형 (12종)
-- ============================================================
INSERT OR IGNORE INTO color_types (code, label_ko, label_en, season, tone, depth, description_ko, description_en, palette_hex) VALUES
('spring_warm_light',  '봄 웜 라이트',  'Spring Warm Light',  'Spring', 'Warm', 'Light',
 '밝고 화사한 복숭아·산호빛 계열이 잘 어울리는 봄 유형입니다.',
 'Peachy, coral, and bright warm hues suit you best.',
 '["#FDDBB4","#F9A87A","#F47C5A","#F9C784","#E8855C"]'),
('spring_warm_bright', '봄 웜 브라이트', 'Spring Warm Bright', 'Spring', 'Warm', 'Bright',
 '선명하고 생기 있는 오렌지·옐로우 계열이 잘 어울립니다.',
 'Vivid orange and yellow tones enhance your natural radiance.',
 '["#FFAA44","#FF7733","#FFD700","#FF9933","#FFBB55"]'),
('summer_cool_light',  '여름 쿨 라이트',  'Summer Cool Light',  'Summer', 'Cool', 'Light',
 '연하고 부드러운 라벤더·베이비핑크 계열이 잘 어울리는 여름 유형입니다.',
 'Soft lavender, baby pink and cool pastels are your best friends.',
 '["#E8D5E8","#C9A7C9","#B8A0C8","#D4B8D4","#E0C8E0"]'),
('summer_cool_muted',  '여름 쿨 뮤트',   'Summer Cool Muted',  'Summer', 'Cool', 'Muted',
 '차분하고 그레이시한 쿨 톤이 우아함을 더해주는 여름 유형입니다.',
 'Muted, greyish cool tones give you an elegant, understated look.',
 '["#B0A8B9","#9890A8","#8888A0","#A0A0B8","#C0B8C8"]'),
('autumn_warm_deep',   '가을 웜 딥',     'Autumn Warm Deep',   'Autumn', 'Warm', 'Deep',
 '깊고 풍부한 테라코타·카키 계열이 잘 어울리는 가을 유형입니다.',
 'Deep terracotta, khaki and rich earth tones bring out your warmth.',
 '["#8B4513","#A0522D","#6B4C2A","#8B6914","#7A4A2A"]'),
('autumn_warm_muted',  '가을 웜 뮤트',   'Autumn Warm Muted',  'Autumn', 'Warm', 'Muted',
 '부드럽고 차분한 올리브·머스터드 계열이 자연스럽게 어우러집니다.',
 'Muted olive, mustard and warm earth tones complement your natural look.',
 '["#B5A642","#9B8B5A","#A09060","#8B7D50","#C4A870"]'),
('winter_cool_deep',   '겨울 쿨 딥',     'Winter Cool Deep',   'Winter', 'Cool', 'Deep',
 '강렬하고 선명한 버건디·네이비·블랙 계열이 잘 어울리는 겨울 유형입니다.',
 'Bold burgundy, navy and stark black make your features stand out.',
 '["#1C1C3A","#2D2D5A","#8B0000","#003366","#2C2C2C"]'),
('winter_cool_bright', '겨울 쿨 브라이트','Winter Cool Bright', 'Winter', 'Cool', 'Bright',
 '선명하고 대비가 강한 로열블루·퓨시아 계열이 잘 어울립니다.',
 'High-contrast royal blue and fuchsia tones suit your striking features.',
 '["#0033CC","#CC0066","#0099CC","#9900CC","#003399"]'),
('spring_warm_deep',   '봄 웜 딥',       'Spring Warm Deep',   'Spring', 'Warm', 'Deep',
 '황금빛이 도는 따뜻하고 깊은 컬러가 잘 어울리는 봄 유형입니다.',
 'Golden warm deep tones give you a rich, sun-kissed glow.',
 '["#C47A1E","#A0602A","#8B5E3C","#B87333","#A0522D"]'),
('summer_cool_bright', '여름 쿨 브라이트','Summer Cool Bright', 'Summer', 'Cool', 'Bright',
 '선명한 로즈·스카이블루 계열이 잘 어울리는 여름 유형입니다.',
 'Vivid rose and sky blue tones brighten your cool complexion.',
 '["#FF69B4","#87CEEB","#FF4499","#66BBEE","#FF66AA"]'),
('autumn_warm_light',  '가을 웜 라이트', 'Autumn Warm Light',  'Autumn', 'Warm', 'Light',
 '따뜻하고 밝은 피치·살구색이 은은하게 어울리는 가을 유형입니다.',
 'Warm, light peach and apricot shades give you a gentle, warm glow.',
 '["#FFCBA4","#FFB87A","#F4A460","#DEB887","#D2A679"]'),
('winter_cool_muted',  '겨울 쿨 뮤트',   'Winter Cool Muted',  'Winter', 'Cool', 'Muted',
 '차갑고 차분한 그레이·플럼 계열이 세련된 분위기를 만듭니다.',
 'Cool grey and plum tones create a sophisticated, understated elegance.',
 '["#708090","#6A5ACD","#7B7B7B","#4A4A6A","#8888AA"]');

-- ============================================================
-- 시드 데이터 — 추천 아이템 (봄 웜 라이트 예시 세트)
-- ============================================================
INSERT OR IGNORE INTO recommended_items
    (color_type_code, category, item_name_ko, item_name_en, color_hex, color_name_ko, color_name_en, tip_ko, tip_en)
VALUES
('spring_warm_light','fashion','피치 블라우스','Peach Blouse','#FFAA88','피치','Peach','가볍고 투명한 소재가 잘 어울려요.','Opt for lightweight, sheer fabrics.'),
('spring_warm_light','fashion','아이보리 원피스','Ivory Dress','#FFFFF0','아이보리','Ivory','과한 장식보다 심플한 실루엣을 선택하세요.','Choose simple silhouettes over heavy embellishments.'),
('spring_warm_light','makeup','코랄 립스틱','Coral Lipstick','#FF6B4A','코랄','Coral','블루 베이스 립은 피부를 칙칙하게 만들어요.','Avoid blue-based reds—they dull your complexion.'),
('spring_warm_light','makeup','피치 블러셔','Peach Blusher','#FFAA77','피치','Peach','광채 마감 제품이 더욱 화사하게 연출돼요.','Choose luminous-finish blushers for extra glow.'),
('spring_warm_light','hair','골든 브라운','Golden Brown','#A0620A','골든 브라운','Golden Brown','너무 어두운 컬러는 얼굴을 무겁게 해요.','Avoid very dark tones—they can look heavy on you.'),
('spring_warm_light','interior','웜 화이트 인테리어','Warm White Interior','#FFFAF0','웜 화이트','Warm White','따뜻한 조명과 함께 사용하면 더욱 포근해요.','Pair with warm-toned lighting for a cozy feel.');
