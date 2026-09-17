# -*- coding: utf-8 -*-
"""
MOCK SERVER — CIB / FaceID / ABS / Payment / SMS хизматларини тақлид қилади.
Фақат ТЕСТ учун. Ҳеч қандай ҳақиқий банк/тўлов операцияси амалга ошмайди.

Ишга тушириш: uvicorn mock_server:app --host 0.0.0.0 --port 9000
"""
import uuid
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

app = FastAPI(title="Mock External Services")


@app.middleware("http")
async def log_requests(request: Request, call_next):
    body = await request.body()
    print(f"[MOCK] {request.method} {request.url.path} -> {body[:500]}")
    response = await call_next(request)
    return response


# ---- CIB (кредит бюроси) ----
@app.post("/api/check")
async def cib_check(request: Request):
    return {"score": 720, "limit": "5000.00"}


# ---- Face ID ----
@app.post("/api/verify")
async def face_verify(request: Request):
    return {"verified": True}


# ---- ABS (банк core тизими) ----
@app.post("/api/contracts")
async def abs_create_contract(request: Request):
    return {"contract_id": f"MOCK-CONTRACT-{uuid.uuid4().hex[:10]}"}


# ---- Payment Gateway ----
@app.post("/api/debit")
async def payment_debit(request: Request):
    return {
        "status": "SUCCESS",
        "transaction_id": f"MOCK-TX-{uuid.uuid4().hex[:10]}",
    }


@app.get("/api/debit/status/{idem}")
async def payment_debit_status(idem: str):
    return {"status": "SUCCESS", "idempotency_key": idem}


# ---- SMS ----
@app.post("/api/send")
async def sms_send(request: Request):
    return {"status": "SENT"}


@app.get("/")
async def root():
    return {"status": "mock server running"}
