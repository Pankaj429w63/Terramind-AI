from __future__ import annotations

from typing import Any, Optional
import json

from backend.supabase_client import get_supabase


class SupabaseRepository:
    """Persistence adapter. A disabled Supabase connection leaves local persistence authoritative."""

    def __init__(self) -> None:
        self.client = get_supabase()

    @property
    def enabled(self) -> bool:
        return self.client is not None

    def save_diagnosis(
        self,
        payload: dict[str, Any],
        user_id: Optional[str] = None,
        image_bytes: Optional[bytes] = None,
        filename: Optional[str] = None,
        content_type: Optional[str] = None,
    ) -> None:
        if not self.client:
            return
        diagnosis = {
            "id": payload["id"],
            "user_id": user_id,
            "source_filename": payload.get("source_filename"),
            "status": "completed",
            "model_info": payload.get("model_info", {}),
            "inference_ms": payload.get("inference_ms"),
            "low_confidence": payload.get("low_confidence", False),
            "low_confidence_threshold": payload.get("low_confidence_threshold"),
            "notes": payload.get("note"),
        }
        self.client.table("diagnoses").upsert(diagnosis).execute()
        predictions = [
            {
                "diagnosis_id": payload["id"],
                "class_id": item["class_id"],
                "label": item["label"],
                "confidence": item.get("confidence", item.get("score", 0)),
                "rank": rank,
            }
            for rank, item in enumerate(payload.get("top5", []), start=1)
        ]
        if predictions:
            self.client.table("predictions").upsert(predictions, on_conflict="diagnosis_id,rank").execute()
        if user_id and image_bytes:
            image_path = f"{user_id}/{payload['id']}/{filename or 'upload.bin'}"
            self.client.storage.from_("diagnosis-images").upload(
                image_path,
                image_bytes,
                {"content-type": content_type or "application/octet-stream", "upsert": "true"},
            )
            self.client.table("uploaded_images").insert({
                "diagnosis_id": payload["id"],
                "user_id": user_id,
                "storage_path": image_path,
                "original_filename": filename,
                "content_type": content_type,
                "byte_size": len(image_bytes),
            }).execute()
        if user_id:
            report_path = f"{user_id}/{payload['id']}.json"
            self.client.storage.from_("diagnosis-reports").upload(
                report_path,
                json.dumps(payload, indent=2).encode("utf-8"),
                {"content-type": "application/json", "upsert": "true"},
            )
            self.client.table("reports").insert({
                "diagnosis_id": payload["id"],
                "user_id": user_id,
                "storage_path": report_path,
                "report_type": "json",
            }).execute()

    def list_history(self, user_id: str, limit: int = 50) -> list[dict[str, Any]]:
        if not self.client:
            return []
        response = (
            self.client.table("diagnoses")
            .select("*, predictions(*)")
            .eq("user_id", user_id)
            .order("created_at", desc=True)
            .limit(limit)
            .execute()
        )
        rows = response.data or []
        for row in rows:
            row["predictions"] = sorted(row.get("predictions", []), key=lambda item: item.get("rank", 999))
            if row["predictions"]:
                first = row["predictions"][0]
                row["prediction"] = {"class_id": first["class_id"], "label": first["label"], "score": first["confidence"], "confidence_pct": round(float(first["confidence"]) * 100, 2)}
                row["predicted_class"] = {**row["prediction"], "confidence": first["confidence"]}
                row["top5"] = [{"class_id": item["class_id"], "label": item["label"], "confidence": item["confidence"], "score": item["confidence"]} for item in row["predictions"]]
            row["timestamp"] = row.get("created_at")
        return rows

    def list_reports(self, user_id: str, limit: int = 50) -> list[dict[str, Any]]:
        if not self.client:
            return []
        response = self.client.table("reports").select("*").eq("user_id", user_id).order("created_at", desc=True).limit(limit).execute()
        return response.data or []

    def record_agent_execution(self, record: dict[str, Any]) -> None:
        if self.client:
            self.client.table("agent_executions").insert(record).execute()

    def save_review(self, record: dict[str, Any]) -> dict[str, Any]:
        if not self.client:
            return record
        response = self.client.table("expert_reviews").insert(record).execute()
        return (response.data or [record])[0]

    def record_chat_event(self, record: dict[str, Any]) -> None:
        if self.client:
            self.client.table("chat_events").insert(record).execute()

    def list_reviews(self, diagnosis_id: str, user_id: str) -> list[dict[str, Any]]:
        if not self.client:
            return []
        response = self.client.table("expert_reviews").select("*").eq("diagnosis_id", diagnosis_id).eq("user_id", user_id).order("created_at", desc=True).execute()
        return response.data or []

    def get_diagnosis(self, diagnosis_id: str, user_id: str) -> dict[str, Any]:
        if not self.client:
            raise LookupError("Supabase unavailable")
        response = self.client.table("diagnoses").select("*, predictions(*), reports(*)").eq("id", diagnosis_id).eq("user_id", user_id).limit(1).execute()
        if not response.data:
            raise LookupError("Diagnosis not found")
        return response.data[0]

    def list_agent_executions(self, limit: int = 50) -> list[dict[str, Any]]:
        if not self.client:
            return []
        response = (
            self.client.table("agent_executions")
            .select("*")
            .order("created_at", desc=True)
            .limit(limit)
            .execute()
        )
        return response.data or []


repository = SupabaseRepository()
