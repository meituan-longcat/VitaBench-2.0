"""Outcome (action-level) reward evaluator for the personalization domain.

Inspects the WRITE tool calls the agent actually made in a subtask, resolves
the booked/ordered entity from the subtask environment, and asks the judge LLM
whether that entity satisfies the rubric. Returns a {0.0, 1.0} reward that the
orchestrator combines with the trajectory reward via min().

Ported from scripts/{ansyc_,}eval_action.py and adapted to use
vita.utils.llm_utils.generate (respecting --evaluator-llm / llm_args_evaluator).
"""

import json
from typing import Any, Dict, List, Optional, Tuple

from loguru import logger

from vita.data_model.message import Message, SystemMessage, UserMessage
from vita.utils.llm_utils import generate


# ─── Constants ───────────────────────────────────────────────────────────────

WRITE_TOOLS = {
    "create_delivery_order",
    "create_instore_product_order",
    "instore_reservation",
    "instore_book",
    "create_hotel_order",
    "create_flight_order",
    "create_train_order",
    "create_attraction_order",
}

# Fields stripped from individual entity dicts shown to the judge
FIELDS_TO_STRIP = {"shop_type", "store_type", "product_type", "distraction_reason", "quantity"}

# Top-level sanitization: fields whose values ("target" / "distraction") leak
# the answer to the judge. room_type is intentionally NOT here — rubrics need
# its real semantic values (大床房, 双床房, ...).
LEAKY_TYPE_FIELDS = {
    "shop_type", "store_type", "product_type",
    "hotel_type", "train_type", "flight_type", "attraction_type",
}
LEAKY_META_FIELDS = {"distraction_reason"}
ENTITY_CONTAINER_KEYS = {
    "stores", "shops", "hotels", "flights", "trains", "attractions",
}


SYSTEM_PROMPT = """\
你是一个行为评估专家。你的任务是判断Agent的实际操作行为是否满足给定的评估标准（rubric）。

你将收到：
1. 评估标准（rubric）：Agent需要满足的所有条件
2. 目标实体（target）：环境中的正确目标实体的完整特征
3. Agent实际选择的实体：Agent通过工具调用实际操作的实体的完整特征

你需要逐条检查rubric中的每个条件，判断Agent实际选择的实体是否满足该条件。

注意事项：
- 只关注Agent的实际行为结果（tool call选择了什么），不关注Agent对话中说了什么
- 对于"预约/预定商家"类操作（instore_reservation），需要判断该商家是否具备rubric要求的服务能力（从其products列表判断）
- 对于"下单商品"类操作（create_delivery_order等），需要判断实际下单的商品是否满足rubric
- 严格按rubric字面要求判断，不能放松标准

输出格式（JSON）：
```json
{
    "rubric_results": [
        {
            "rubric": "<rubric条件原文>",
            "met": true/false,
            "justification": "<简要解释>"
        }
    ],
    "all_met": true/false,
    "action_reward": 1.0 或 0.0
}
```
"""

USER_PROMPT_TEMPLATE = """\
# 评估标准（Rubric）
{rubric_json}

# 目标实体（环境中的正确答案）
{target_json}

# Agent实际选择的实体
操作类型: {action_type}
{entity_json}

# 请评估Agent的实际操作是否满足所有rubric条件
"""


# ─── Env Sanitization ────────────────────────────────────────────────────────


def _sanitize_entity(obj: Any) -> Any:
    """Recursively drop leakage fields from an entity subtree.

    Preserves target identity: if any *_type value equals "target" on this
    dict, re-inject `shop_type: "target"` so find_target_in_env() can still
    locate it. All distraction markers are dropped.
    """
    if isinstance(obj, dict):
        is_target = any(obj.get(k) == "target" for k in LEAKY_TYPE_FIELDS)
        cleaned = {}
        for k, v in obj.items():
            if k in LEAKY_TYPE_FIELDS or k in LEAKY_META_FIELDS:
                continue
            cleaned[k] = _sanitize_entity(v)
        if is_target:
            cleaned["shop_type"] = "target"
        return cleaned
    if isinstance(obj, list):
        return [_sanitize_entity(x) for x in obj]
    return obj


def sanitize_env(env: dict) -> dict:
    """Deep-clean one subtask's environment before feeding downstream."""
    if not isinstance(env, dict):
        return env
    out = {}
    for key, container in env.items():
        if key not in ENTITY_CONTAINER_KEYS:
            out[key] = container
            continue
        if isinstance(container, dict):
            out[key] = {eid: _sanitize_entity(v) for eid, v in container.items()}
        elif isinstance(container, list):
            out[key] = [_sanitize_entity(v) for v in container]
        else:
            out[key] = container
    return out


# ─── Env Lookups ─────────────────────────────────────────────────────────────


def strip_meta_fields(obj: dict) -> dict:
    return {k: v for k, v in obj.items() if k not in FIELDS_TO_STRIP}


def find_store_in_env(env: dict, store_id: str) -> Optional[dict]:
    for key in ["stores", "shops"]:
        container = env.get(key, {})
        if isinstance(container, dict):
            if store_id in container:
                return container[store_id]
            for sid, store in container.items():
                if store.get("store_id") == store_id or sid == store_id:
                    return store
        elif isinstance(container, list):
            for store in container:
                if store.get("store_id") == store_id or store.get("shop_id") == store_id:
                    return store
    return None


def find_product_in_store(store: dict, product_id: str) -> Optional[dict]:
    for p in store.get("products", []):
        if p.get("product_id") == product_id:
            return p
    return None


def find_hotel_in_env(env: dict, hotel_id: str) -> Optional[dict]:
    hotels = env.get("hotels", {})
    if isinstance(hotels, dict):
        if hotel_id in hotels:
            return hotels[hotel_id]
    elif isinstance(hotels, list):
        for h in hotels:
            if h.get("hotel_id") == hotel_id:
                return h
    return None


def find_room_in_hotel(hotel: dict, room_id: str) -> Optional[dict]:
    rooms = hotel.get("rooms", hotel.get("products", []))
    if isinstance(rooms, list):
        for r in rooms:
            if r.get("room_id") == room_id or r.get("product_id") == room_id:
                return r
    return None


def find_entity_in_env(env: dict, entity_type: str, entity_id: str) -> Optional[dict]:
    container = env.get(entity_type, {})
    if isinstance(container, dict):
        if entity_id in container:
            return container[entity_id]
    elif isinstance(container, list):
        for item in container:
            for id_field in [f"{entity_type[:-1]}_id", "id", "product_id"]:
                if item.get(id_field) == entity_id:
                    return item
    return None


def find_ticket_in_entity(entity: dict, ticket_id: str) -> Optional[dict]:
    for key in ["tickets", "seats", "products"]:
        items = entity.get(key, [])
        if isinstance(items, list):
            for item in items:
                for id_field in ["ticket_id", "seat_id", "product_id"]:
                    if item.get(id_field) == ticket_id:
                        return item
    return None


def find_target_in_env(env: dict) -> Optional[dict]:
    """Find the target entity (marked by *_type == 'target' after sanitization)."""
    for key in ["stores", "shops", "hotels", "flights", "trains", "attractions"]:
        container = env.get(key, {})
        if isinstance(container, dict):
            for eid, entity in container.items():
                if entity.get("shop_type") == "target" or entity.get("store_type") == "target":
                    clean = strip_meta_fields(entity)
                    clean_products = []
                    for p in entity.get("products", entity.get("rooms", [])):
                        if isinstance(p, dict) and p.get("product_type") == "target":
                            clean_products.append(strip_meta_fields(p))
                    if clean_products:
                        clean["target_products"] = clean_products
                    return clean
        elif isinstance(container, list):
            for entity in container:
                if entity.get("shop_type") == "target" or entity.get("store_type") == "target":
                    return strip_meta_fields(entity)
    return None


# ─── Action Resolution ───────────────────────────────────────────────────────


def resolve_action(tool_name: str, args: dict, env: dict) -> Optional[dict]:
    """Resolve a WRITE tool call to the actual entity features from env.

    Returns {action_type, entity_info, product_info, extra} or None if the
    tool name is unknown.
    """
    result = {"action_type": tool_name, "entity_info": None, "product_info": None}

    if tool_name == "create_delivery_order":
        store_id = args.get("store_id", "")
        product_ids = args.get("product_ids", args.get("food_ids", []))
        store = find_store_in_env(env, store_id)
        if store:
            result["entity_info"] = strip_meta_fields(
                {k: v for k, v in store.items() if k != "products"}
            )
            products_found = []
            for pid in product_ids:
                p = find_product_in_store(store, pid)
                if p:
                    products_found.append(strip_meta_fields(p))
            result["product_info"] = products_found
            result["extra"] = {
                "address": args.get("address", ""),
                "dispatch_time": args.get("dispatch_time", ""),
                "attributes": args.get("attributes", ""),
                "note": args.get("note", ""),
            }
        return result

    elif tool_name == "create_instore_product_order":
        shop_id = args.get("shop_id", "")
        product_id = args.get("product_id", "")
        store = find_store_in_env(env, shop_id)
        if store:
            result["entity_info"] = strip_meta_fields(
                {k: v for k, v in store.items() if k != "products"}
            )
            p = find_product_in_store(store, product_id)
            if p:
                result["product_info"] = [strip_meta_fields(p)]
        result["extra"] = {"order_quantity": args.get("quantity", 1)}
        return result

    elif tool_name in ("instore_reservation", "instore_book"):
        shop_id = args.get("shop_id", "")
        store = find_store_in_env(env, shop_id)
        if store:
            result["entity_info"] = strip_meta_fields(
                {k: v for k, v in store.items() if k != "products"}
            )
            all_products = [strip_meta_fields(p) for p in store.get("products", [])]
            result["product_info"] = all_products
            result["extra"] = {
                "time": args.get("time", ""),
                "customer_count": args.get("customer_count", ""),
            }
        return result

    elif tool_name == "create_hotel_order":
        hotel_id = args.get("hotel_id", "")
        room_id = args.get("room_id", "")
        hotel = find_hotel_in_env(env, hotel_id)
        if hotel:
            result["entity_info"] = strip_meta_fields(
                {k: v for k, v in hotel.items() if k not in ("rooms", "products")}
            )
            room = find_room_in_hotel(hotel, room_id)
            if room:
                result["product_info"] = [strip_meta_fields(room)]
        return result

    elif tool_name in ("create_flight_order", "create_train_order"):
        entity_type = "flights" if "flight" in tool_name else "trains"
        id_key = "flight_id" if "flight" in tool_name else "train_id"
        entity_id = args.get(id_key, "")
        seat_id = args.get("seat_id", "")
        entity = find_entity_in_env(env, entity_type, entity_id)
        if entity:
            result["entity_info"] = strip_meta_fields(
                {k: v for k, v in entity.items() if k not in ("seats", "products")}
            )
            seat = find_ticket_in_entity(entity, seat_id)
            if seat:
                result["product_info"] = [strip_meta_fields(seat)]
        result["extra"] = {"date": args.get("date", ""), "quantity": args.get("quantity", "")}
        return result

    elif tool_name == "create_attraction_order":
        attraction_id = args.get("attraction_id", "")
        ticket_id = args.get("ticket_id", "")
        entity = find_entity_in_env(env, "attractions", attraction_id)
        if entity:
            result["entity_info"] = strip_meta_fields(
                {k: v for k, v in entity.items() if k not in ("tickets", "products")}
            )
            ticket = find_ticket_in_entity(entity, ticket_id)
            if ticket:
                result["product_info"] = [strip_meta_fields(ticket)]
        result["extra"] = {"date": args.get("date", ""), "quantity": args.get("quantity", "")}
        return result

    return None


# ─── Write-action Extraction from Pydantic Messages ──────────────────────────


def extract_write_actions(messages: List[Message]) -> List[Tuple[str, dict]]:
    """Collect (tool_name, args) for every WRITE tool call in the subtask.

    Works on pydantic Message objects (unlike the dict-based version in
    scripts/eval_action.py). Messages without tool_calls (SystemMessage,
    ToolMessage, MultiToolMessage) are silently ignored.
    """
    actions: List[Tuple[str, dict]] = []
    for msg in messages:
        tool_calls = getattr(msg, "tool_calls", None)
        if not tool_calls:
            continue
        for tc in tool_calls:
            name = getattr(tc, "name", None)
            if name in WRITE_TOOLS:
                args = getattr(tc, "arguments", None) or {}
                actions.append((name, args))
    return actions


# ─── Public API ──────────────────────────────────────────────────────────────


def evaluate_outcome_for_subtask(
    rubric: List[str],
    env: dict,
    messages: List[Message],
    llm_evaluator: str,
    llm_args_evaluator: Optional[dict] = None,
) -> Dict[str, Any]:
    """Evaluate a single subtask's action reward.

    Args:
        rubric: The overall_rubrics list for this subtask.
        env: The raw subtask environment (sanitized internally).
        messages: This subtask's message list (post-split by the orchestrator).
        llm_evaluator: Model name for the judge (e.g. "gpt-4.1").
        llm_args_evaluator: Optional kwargs forwarded to generate().

    Returns:
        Dict with keys: has_action, action_reward, action_type, llm_result, note.
    """
    llm_args_evaluator = llm_args_evaluator or {}
    clean_env = sanitize_env(env)
    actions = extract_write_actions(messages)

    if not actions:
        return {
            "has_action": False,
            "action_reward": None,
            "action_type": None,
            "llm_result": {},
            "note": "No WRITE tool calls in this subtask",
        }

    resolved = []
    for tool_name, args in actions:
        r = resolve_action(tool_name, args, clean_env)
        if r and r.get("entity_info"):
            resolved.append(r)

    if not resolved:
        return {
            "has_action": True,
            "action_reward": 0.0,
            "action_type": actions[-1][0],
            "llm_result": {},
            "note": "WRITE tool calls found but could not resolve entities in env",
        }

    last = resolved[-1]
    target = find_target_in_env(clean_env)

    entity_desc = f"商家/实体信息: {json.dumps(last['entity_info'], ensure_ascii=False, indent=2)}"
    if last.get("product_info"):
        entity_desc += f"\n商品/产品信息: {json.dumps(last['product_info'], ensure_ascii=False, indent=2)}"
    if last.get("extra"):
        entity_desc += f"\n附加参数: {json.dumps(last['extra'], ensure_ascii=False, indent=2)}"

    user_prompt = USER_PROMPT_TEMPLATE.format(
        rubric_json=json.dumps(rubric, ensure_ascii=False, indent=2),
        target_json=(
            json.dumps(target, ensure_ascii=False, indent=2) if target else "未找到target实体"
        ),
        action_type=last["action_type"],
        entity_json=entity_desc,
    )

    try:
        response = generate(
            model=llm_evaluator,
            messages=[
                SystemMessage(role="system", content=SYSTEM_PROMPT),
                UserMessage(role="user", content=user_prompt),
            ],
            **llm_args_evaluator,
        )
        raw = response.content or ""
    except Exception as e:
        logger.warning(f"Outcome judge LLM call failed: {e}")
        return {
            "has_action": True,
            "action_reward": 0.0,
            "action_type": last["action_type"],
            "llm_result": {"error": str(e)},
            "note": "Judge LLM call failed",
        }

    try:
        clean = raw.strip()
        if clean.startswith("```"):
            clean = clean.split("\n", 1)[1]
            clean = clean.rsplit("```", 1)[0]
        parsed = json.loads(clean)
        action_reward = float(parsed.get("action_reward", 0.0))
    except (json.JSONDecodeError, IndexError, ValueError):
        logger.warning("Outcome judge returned unparseable JSON; defaulting to 0.0")
        parsed = {"raw_response": raw}
        action_reward = 0.0

    return {
        "has_action": True,
        "action_reward": action_reward,
        "action_type": last["action_type"],
        "llm_result": parsed,
        "note": None,
    }
