import type { ModelInfo, ResumeAnalysisResponse } from '../types'

export const API_BASE_URL = (
  import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:8000/api/v1'
).replace(/\/$/, '')

export class ApiError extends Error {
  constructor(
    message: string,
    public readonly status?: number,
    public readonly requestId?: string,
  ) {
    super(message)
    this.name = 'ApiError'
  }
}

async function parseError(response: Response): Promise<ApiError> {
  const requestId = response.headers.get('x-request-id') ?? undefined
  try {
    const payload = (await response.json()) as {
      detail?: string | { message?: string }
      error?: { message?: string; request_id?: string }
    }
    const detail = payload.detail
    const message = payload.error?.message || (typeof detail === 'string' ? detail : detail?.message)
    return new ApiError(
      message || `Request failed (${response.status})`,
      response.status,
      payload.error?.request_id || requestId,
    )
  } catch {
    return new ApiError(`Request failed (${response.status})`, response.status, requestId)
  }
}

async function request<T>(path: string, options?: RequestInit): Promise<T> {
  let response: Response
  try {
    response = await fetch(`${API_BASE_URL}${path}`, options)
  } catch {
    throw new ApiError('Cannot reach the analysis service. Confirm the FastAPI server is running.')
  }
  if (!response.ok) throw await parseError(response)
  return response.json() as Promise<T>
}

export function getModelInfo(signal?: AbortSignal): Promise<ModelInfo> {
  return request<ModelInfo>('/model', { signal })
}

export function analyzeResume(
  file: File,
  topK: number,
  signal?: AbortSignal,
): Promise<ResumeAnalysisResponse> {
  const body = new FormData()
  body.append('file', file)
  return request<ResumeAnalysisResponse>(`/analyze-resume?top_k=${topK}`, {
    method: 'POST',
    body,
    signal,
  })
}
