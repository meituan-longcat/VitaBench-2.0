"""
Shared fixtures for memory module tests.

Provides realistic interaction data mimicking the vitabench personalize domain.
Two formats are supported:
  1. init_gen format: {date, behavior: [...], dialogue: [...]}
  2. Interaction model format: {type, timestamp, content}
"""

import pytest


# ──────────────────────────────────────────────────────────────────────
# init_gen 格式的交互数据 (tasks.json 中真实使用的格式)
# ──────────────────────────────────────────────────────────────────────

@pytest.fixture
def delivery_interactions_initgen():
    """外卖场景: 高中生日常点外卖, 偏好川菜盖饭、大份量、价格实惠."""
    return [
        {
            "date": "2023-03-10",
            "behavior": [
                {
                    "behavior_type": "search",
                    "content": {"keyword": "红烧肉盖饭"},
                },
                {
                    "behavior_type": "order",
                    "content": {
                        "scenario": "delivery",
                        "merchant_name": "川味盖饭王(大邑晋原店)",
                        "tags": ["川菜", "盖饭", "大份量", "实惠"],
                        "items": [
                            {
                                "product_name": "红烧肉盖饭(大份)",
                                "price": 18.0,
                                "quantity": 1,
                            },
                            {
                                "product_name": "可乐",
                                "price": 3.0,
                                "quantity": 1,
                            },
                        ],
                    },
                },
            ],
            "dialogue": [
                {"role": "user", "content": "帮我点个外卖，要有肉的盖饭，大份的"},
                {"role": "assistant", "content": "好的，我看到附近有川味盖饭王，他们的红烧肉盖饭大份18元，份量很足"},
                {"role": "user", "content": "行，就这个吧，再加个可乐"},
            ],
        },
        {
            "date": "2023-03-15",
            "behavior": [
                {
                    "behavior_type": "search",
                    "content": {"keyword": "回锅肉"},
                },
                {
                    "behavior_type": "high_freq_browse",
                    "content": {"topic": "川菜盖饭店铺"},
                },
                {
                    "behavior_type": "order",
                    "content": {
                        "scenario": "delivery",
                        "merchant_name": "川味盖饭王(大邑晋原店)",
                        "tags": ["川菜", "回锅肉", "特大份"],
                        "items": [
                            {
                                "product_name": "回锅肉盖饭(特大份)",
                                "price": 22.0,
                                "quantity": 1,
                            },
                        ],
                    },
                },
            ],
            "dialogue": [
                {"role": "user", "content": "又想吃川味盖饭王了，换个口味"},
                {"role": "assistant", "content": "回锅肉盖饭特大份怎么样？22块，600g超大份量"},
                {"role": "user", "content": "可以可以，就这个"},
            ],
        },
    ]


@pytest.fixture
def instore_interactions_initgen():
    """到店消费场景: 学生和朋友去吃火锅, 偏好微辣、鸳鸯锅."""
    return [
        {
            "date": "2023-04-20",
            "behavior": [
                {
                    "behavior_type": "search",
                    "content": {"keyword": "火锅 大邑"},
                },
                {
                    "behavior_type": "browse",
                    "content": {"topic": "大邑县火锅店推荐"},
                },
                {
                    "behavior_type": "order",
                    "content": {
                        "scenario": "instore",
                        "merchant_name": "蜀九香火锅(大邑店)",
                        "tags": ["火锅", "川味", "鸳鸯锅"],
                        "items": [
                            {
                                "product_name": "鸳鸯锅底",
                                "price": 68.0,
                                "quantity": 1,
                            },
                            {
                                "product_name": "毛肚",
                                "price": 38.0,
                                "quantity": 1,
                            },
                            {
                                "product_name": "鸭肠",
                                "price": 28.0,
                                "quantity": 1,
                            },
                        ],
                    },
                },
                {
                    "behavior_type": "review",
                    "content": {
                        "rating": 4,
                        "text": "鸳鸯锅很不错，微辣那边味道很好，毛肚很新鲜。就是周末人太多等位久。",
                    },
                },
            ],
            "dialogue": [
                {"role": "user", "content": "想和同学一起去吃火锅，有没有推荐的"},
                {"role": "assistant", "content": "蜀九香火锅在大邑评分挺高的，有鸳鸯锅可以选"},
                {"role": "user", "content": "好，要鸳鸯锅，一边微辣一边不辣，我有个同学不吃辣"},
            ],
        },
    ]


@pytest.fixture
def ota_interactions_initgen():
    """酒旅场景: 学生暑假出行, 预算有限偏好青旅/经济酒店."""
    return [
        {
            "date": "2023-07-01",
            "behavior": [
                {
                    "behavior_type": "search",
                    "content": {"keyword": "成都青年旅舍"},
                },
                {
                    "behavior_type": "high_freq_browse",
                    "content": {"topic": "成都市区经济型住宿"},
                },
                {
                    "behavior_type": "order",
                    "content": {
                        "scenario": "ota",
                        "merchant_name": "梦之旅青年旅舍(春熙路店)",
                        "tags": ["青旅", "经济", "市中心", "交通便利"],
                        "items": [
                            {
                                "product_name": "四人间床位(男生)",
                                "price": 58.0,
                                "quantity": 2,
                            },
                        ],
                    },
                },
            ],
            "dialogue": [
                {"role": "user", "content": "暑假想去成都市区逛逛，住两晚，预算低一点"},
                {"role": "assistant", "content": "春熙路附近有个梦之旅青年旅舍，四人间58一晚，离地铁很近"},
                {"role": "user", "content": "可以，帮我订两晚"},
            ],
        },
        {
            "date": "2023-07-05",
            "behavior": [
                {
                    "behavior_type": "search",
                    "content": {"keyword": "都江堰一日游"},
                },
                {
                    "behavior_type": "order",
                    "content": {
                        "scenario": "ota",
                        "merchant_name": "携程旅游",
                        "tags": ["一日游", "都江堰", "学生票"],
                        "items": [
                            {
                                "product_name": "都江堰-青城山一日游(学生票)",
                                "price": 128.0,
                                "quantity": 2,
                            },
                        ],
                    },
                },
            ],
            "dialogue": [],
        },
    ]


@pytest.fixture
def pharmacy_interactions_initgen():
    """药品下单场景: 用户过敏性鼻炎发作, 急需购药."""
    return [
        {
            "date": "2023-07-21",
            "behavior": [
                {
                    "behavior_type": "search",
                    "content": {"keyword": "氯雷他定片"},
                },
                {
                    "behavior_type": "order",
                    "content": {
                        "scenario": "delivery",
                        "merchant_name": "康佳大药房(大邑店)",
                        "tags": ["药品", "快速配送"],
                        "items": [
                            {
                                "product_name": "氯雷他定片 12片",
                                "price": 22.0,
                                "quantity": 1,
                            },
                            {
                                "product_name": "生理盐水鼻腔喷雾",
                                "price": 35.0,
                                "quantity": 1,
                            },
                        ],
                    },
                },
            ],
            "dialogue": [
                {"role": "user", "content": "鼻炎又犯了，帮我买点药"},
                {"role": "assistant", "content": "了解了，康佳大药房有氯雷他定片和鼻腔喷雾，可以快速配送"},
                {"role": "user", "content": "都要吧，尽快送到"},
            ],
        },
    ]


# ──────────────────────────────────────────────────────────────────────
# Interaction model 格式 ({type, timestamp, content})
# ──────────────────────────────────────────────────────────────────────

@pytest.fixture
def delivery_interactions_model():
    """Interaction model 格式的外卖交互."""
    return [
        {
            "type": "search",
            "timestamp": "2023-03-10 11:30:00",
            "content": {"keyword": "红烧肉盖饭"},
        },
        {
            "type": "order",
            "timestamp": "2023-03-10 12:05:00",
            "content": {
                "scenario": "delivery",
                "merchant_name": "川味盖饭王(大邑晋原店)",
                "items": [
                    {"product_name": "红烧肉盖饭(大份)", "price": 18.0, "quantity": 1},
                ],
            },
        },
        {
            "type": "review",
            "timestamp": "2023-03-10 13:00:00",
            "content": {
                "rating": 5,
                "text": "份量超足，肉很多很下饭，以后就吃这家了",
            },
        },
    ]


@pytest.fixture
def browse_and_cart_interactions_model():
    """浏览加购场景: 用户反复浏览但未下单."""
    return [
        {
            "type": "browse",
            "timestamp": "2023-04-05 18:00:00",
            "content": {"topic": "华莱士新品鸡腿堡"},
        },
        {
            "type": "add_to_cart",
            "timestamp": "2023-04-05 18:10:00",
            "content": {
                "merchant_name": "华莱士(大邑店)",
                "items": [
                    {"product_name": "香辣鸡腿堡套餐", "price": 18.9, "quantity": 1},
                ],
            },
        },
        {
            "type": "favorite",
            "timestamp": "2023-04-05 18:12:00",
            "content": {
                "merchant_name": "华莱士(大邑店)",
                "type": "store",
            },
        },
    ]


@pytest.fixture
def complaint_interaction_model():
    """投诉场景: 用户对配送不满."""
    return [
        {
            "type": "complaint",
            "timestamp": "2023-05-12 20:30:00",
            "content": {
                "merchant_name": "老王烧烤(大邑店)",
                "reason": "送达超时30分钟，烧烤都凉了",
                "resolution": "退款50%",
            },
        },
    ]


# ──────────────────────────────────────────────────────────────────────
# 多轮连续交互 (模拟一个用户跨多个 subtask 的完整 memory 演进)
# ──────────────────────────────────────────────────────────────────────

@pytest.fixture
def multi_turn_interactions():
    """模拟三轮 subtask 的交互序列, 每轮更新 memory 后进入下一轮.

    Turn 1: 初始外卖点餐 (建立初始偏好)
    Turn 2: 再次点餐 (偏好得到强化)
    Turn 3: 酒旅出行 (新维度偏好)
    """
    turn1 = [
        {
            "date": "2023-03-10",
            "behavior": [
                {
                    "behavior_type": "order",
                    "content": {
                        "scenario": "delivery",
                        "merchant_name": "川味盖饭王",
                        "tags": ["川菜", "大份量"],
                        "items": [{"product_name": "红烧肉盖饭(大份)", "price": 18.0, "quantity": 1}],
                    },
                },
            ],
            "dialogue": [
                {"role": "user", "content": "帮我点个大份盖饭，要有肉的"},
                {"role": "assistant", "content": "好的，红烧肉盖饭大份18元"},
            ],
        },
    ]
    turn2 = [
        {
            "date": "2023-03-20",
            "behavior": [
                {
                    "behavior_type": "order",
                    "content": {
                        "scenario": "delivery",
                        "merchant_name": "川味盖饭王",
                        "tags": ["川菜", "回锅肉"],
                        "items": [{"product_name": "回锅肉盖饭(特大份)", "price": 22.0, "quantity": 1}],
                    },
                },
                {
                    "behavior_type": "review",
                    "content": {"rating": 5, "text": "回锅肉特别正宗，每次都来这家"},
                },
            ],
            "dialogue": [],
        },
    ]
    turn3 = [
        {
            "date": "2023-07-01",
            "behavior": [
                {
                    "behavior_type": "order",
                    "content": {
                        "scenario": "ota",
                        "merchant_name": "梦之旅青年旅舍",
                        "tags": ["青旅", "经济"],
                        "items": [{"product_name": "四人间床位", "price": 58.0, "quantity": 2}],
                    },
                },
            ],
            "dialogue": [
                {"role": "user", "content": "暑假想住便宜点的"},
            ],
        },
    ]
    return [turn1, turn2, turn3]


# ──────────────────────────────────────────────────────────────────────
# 用户画像 fixture (来自 tasks.json 的真实格式)
# ──────────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_user_profile():
    """高中生用户画像, 来自真实 tasks.json."""
    return {
        "user_id": "U901652",
        "职业": "高中生",
        "性别": "男",
        "出生日期": "2010-08-18",
        "常住地": "成都",
        "常住住址": "四川省成都市大邑县兴隆佳苑33栋2单元3号",
        "工作地址": "四川省成都市大邑县安仁中学",
        "家庭情况": ["单身", "有一个姐姐"],
        "籍贯": "四川成都",
        "学历": "高中",
        "长期疾病/过敏史": "过敏性鼻炎",
    }


# ──────────────────────────────────────────────────────────────────────
# 空交互 / 边界情况
# ──────────────────────────────────────────────────────────────────────

@pytest.fixture
def empty_interactions():
    return []


@pytest.fixture
def single_behavior_no_dialogue():
    """只有行为没有对话的交互."""
    return [
        {
            "date": "2023-06-01",
            "behavior": [
                {
                    "behavior_type": "search",
                    "content": {"keyword": "塔斯汀汉堡"},
                },
            ],
        },
    ]


@pytest.fixture
def dialogue_only_no_behavior():
    """只有对话没有行为的交互."""
    return [
        {
            "date": "2023-06-02",
            "dialogue": [
                {"role": "user", "content": "最近有什么优惠活动吗"},
                {"role": "assistant", "content": "目前华莱士有买一送一活动"},
            ],
        },
    ]
