"use client";

import { useCallback, useState } from "react";
import { Diagnosis, predictBatch, predictImage } from "../services/diagnosis";

export function useDiagnosis() {
  const [result, setResult] = useState<Diagnosis | Diagnosis[] | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const run = useCallback(async (files: File[], batch = false) => {
    if (!files.length) {
      setError("Choose at least one image.");
      return;
    }
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      if (batch) {
        const response = await predictBatch(files);
        const failures = response.results.filter((item) => item.error);
        if (failures.length === response.results.length) throw new Error(failures[0]?.error || "Batch prediction failed");
        setResult(response.results.flatMap((item) => item.prediction ? [item.prediction] : []));
      } else {
        setResult(await predictImage(files[0]));
      }
    } catch (requestError) {
      setError(requestError instanceof Error ? requestError.message : "Unable to reach the inference API.");
    } finally {
      setLoading(false);
    }
  }, []);

  return { result, loading, error, run, clear: () => { setResult(null); setError(null); } };
}
