# # app/services/extract.py
# from typing import Any, Dict, List
# import os
# import json
# import re
# from pydantic import ValidationError
# from dotenv import load_dotenv
# import inspect
# print("🧠 LOADED extract.py FROM:", __file__)
#
# load_dotenv()
#
# from app.schemas.models import Line, Mod
#
#
# # -------------------------
# # Helpers
# # -------------------------
# def _best_effort_parse_cart(obj: Any) -> List[Line]:
#     out: List[Line] = []
#     if not obj:
#         return out
#     if isinstance(obj, dict) and obj.get("lines"):
#         obj = obj.get("lines")
#     if not isinstance(obj, list):
#         return out
#     for raw in obj:
#         try:
#             if isinstance(raw, dict) and "modifiers" in raw and isinstance(raw["modifiers"], list):
#                 raw["modifiers"] = [Mod(**m) for m in raw["modifiers"]]
#             out.append(Line(**raw))
#         except ValidationError:
#             continue
#     return out
#
#
# def _parse_json_loose(text: str) -> Dict[str, Any]:
#     """
#     Last-resort JSON extractor: grabs the first top-level {...} block.
#     """
#     if not text:
#         return {}
#     # Find first '{' and last '}' to approximate a JSON object
#     start = text.find("{")
#     end = text.rfind("}")
#     if start == -1 or end == -1 or end <= start:
#         return {}
#     snippet = text[start : end + 1]
#     try:
#         return json.loads(snippet)
#     except Exception:
#         # try to remove trailing commas etc.
#         snippet = re.sub(r",\s*([}\]])", r"\1", snippet)
#         try:
#             return json.loads(snippet)
#         except Exception:
#             return {}
#
#
# # -------------------------
# # LLM extractor (Chat Completions only; resilient)
# # -------------------------
# def extract_cart_with_llm(transcript: str, collected: Dict[str, Any]) -> Dict[str, Any]:
#     """
#     Extract structured cart lines from transcript using OpenAI Chat Completions
#     with JSON Schema -> JSON Object -> Loose JSON fallbacks.
#     """
#     print("🧪 RUNNING extract_cart_with_llm FROM:", inspect.getsourcefile(extract_cart_with_llm))
#
#     print("🧠 Extractor version: CHAT_COMPLETIONS_FALLBACK running!")
#
#     # 0) If upstream already supplied structure, use it
#     structured = collected.get("structured_order") or {}
#     lines = _best_effort_parse_cart(structured)
#     if lines:
#         return {
#             "cart_lines": lines,
#             "notes": structured.get("notes") or "",
#             "detected_menu": structured.get("menu_hint"),
#         }
#
#     api_key = os.getenv("OPENAI_API_KEY")
#     if not api_key:
#         return {
#             "cart_lines": [],
#             "notes": "",
#             "detected_menu": None,
#             "reason": "LLM disabled: OPENAI_API_KEY not set."
#         }
#
#     # JSON schema the model should follow
#     schema: Dict[str, Any] = {
#         "name": "OrderExtraction",
#         "strict": True,
#         "schema": {
#             "type": "object",
#             "properties": {
#                 "cart_lines": {
#                     "type": "array",
#                     "items": {
#                         "type": "object",
#                         "additionalProperties": False,
#                         "properties": {
#                             "item_id": {"type": "string", "minLength": 1},
#                             "qty": {"type": "integer", "minimum": 1},
#                             "variant_id": {"type": "string"},
#                             "combo_opt_in": {"type": "boolean", "default": False},
#                             "modifiers": {
#                                 "type": "array",
#                                 "items": {
#                                     "type": "object",
#                                     "additionalProperties": False,
#                                     "properties": {
#                                         "modifier_id": {"type": "string", "minLength": 1},
#                                         "qty": {"type": "integer", "minimum": 1, "default": 1},
#                                         "unit_price": {"type": "number"}
#                                     },
#                                     "required": ["modifier_id"]
#                                 },
#                                 "default": []
#                             },
#                             "attributes": {
#                                 "type": "object",
#                                 "additionalProperties": True,
#                                 "default": {}
#                             },
#                             "utterance": {"type": "string", "default": ""},
#                             "menu_hint": {
#                                 "type": "string",
#                                 "enum": ["american", "middle-eastern"],
#                             }
#                         },
#                         "required": ["item_id", "qty"]
#                     },
#                     "default": []
#                 },
#                 "notes": {"type": "string", "default": ""},
#                 "detected_menu": {
#                     "type": ["string", "null"],
#                     "enum": ["american", "middle-eastern", None],
#                     "default": None
#                 },
#                 "reason": {"type": "string", "default": ""}
#             },
#             "required": ["cart_lines", "notes", "detected_menu", "reason"],
#             "additionalProperties": False
#         }
#     }
#
#     system_msg = (
#         "You extract a takeaway food order from a phone call transcript. "
#         "Output STRICTLY as JSON matching the provided schema. "
#         "If no valid items, return empty cart_lines and explain in 'reason'. "
#         "Do not include any text outside the JSON object."
#     )
#     tenant_hint = collected.get("menu_hint") or collected.get("tenant") or ""
#     user_msg = f"""Transcript:
# {transcript}
#
# Guidelines:
# - Use canonical ids (e.g., 'chicken-shawarma', 'extra-chicken', etc.)
# - If 'combo' is mentioned and the category allows combos, set combo_opt_in=true
# - Put rice types etc. into attributes (attributes.rice_type='yellow')
# - If you can infer menu, set detected_menu to 'american' or 'middle-eastern'
# - If unsure or no order, leave cart_lines=[] and explain in 'reason'
# Tenant hint: {tenant_hint or "none"}"""
#
#     from openai import OpenAI
#     client = OpenAI(api_key=api_key)
#     model = os.getenv("EXTRACTOR_MODEL", "gpt-4o-mini")
#
#     # Try JSON Schema (if SDK supports it on chat.completions), else json_object, else plain text
#     data: Dict[str, Any] = {}
#
#     try:
#         # Attempt Chat Completions + JSON Schema (newer SDKs)
#         try:
#             rsp = client.chat.completions.create(
#                 model=model,
#                 messages=[
#                     {"role": "system", "content": system_msg},
#                     {"role": "user", "content": user_msg},
#                 ],
#                 temperature=0.1,
#                 response_format={"type": "json_schema", "json_schema": schema},
#             )
#             txt = rsp.choices[0].message.content or "{}"
#             data = json.loads(txt)
#         except TypeError:
#             # Fallback: Chat Completions with json_object mode (widely supported)
#             rsp = client.chat.completions.create(
#                 model=model,
#                 messages=[
#                     {"role": "system", "content": system_msg},
#                     {"role": "user", "content": user_msg + "\nReturn ONLY a JSON object per schema."},
#                 ],
#                 temperature=0.1,
#                 response_format={"type": "json_object"},
#             )
#             txt = rsp.choices[0].message.content or "{}"
#             data = json.loads(txt)
#
#     except Exception:
#         # Last resort: plain text (no response_format). We’ll parse the JSON block.
#         try:
#             rsp = client.chat.completions.create(
#                 model=model,
#                 messages=[
#                     {"role": "system", "content": system_msg},
#                     {"role": "user", "content": user_msg + "\nReturn ONLY a JSON object matching the schema. No extra text."},
#                 ],
#                 temperature=0.1,
#             )
#             txt = rsp.choices[0].message.content or ""
#             data = _parse_json_loose(txt)
#         except Exception as e:
#             return {
#                 "cart_lines": [],
#                 "notes": "",
#                 "detected_menu": None,
#                 "reason": f"LLM error: {e}"
#             }
#
#     # Coerce into your models
#     cart_lines = _best_effort_parse_cart(data.get("cart_lines"))
#     notes = data.get("notes", "")
#     detected_menu = data.get("detected_menu")
#
#     if not cart_lines:
#         return {
#             "cart_lines": [],
#             "notes": notes or "",
#             "detected_menu": detected_menu,
#             "reason": data.get("reason") or "No actionable items/quantities recognized."
#         }
#
#     # Backfill useful fields
#     for ln in cart_lines:
#         if not getattr(ln, "utterance", None):
#             ln.utterance = transcript
#         if detected_menu and not getattr(ln, "menu_hint", None):
#             ln.menu_hint = detected_menu
#
#     return {
#         "cart_lines": cart_lines,
#         "notes": notes or "",
#         "detected_menu": detected_menu,
#         "reason": ""
#     }
