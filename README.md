# hybrid-entity-resolution
A Multi-stage retrieval pipeline for financial reconciliation using BGE-M3 (Bi-Encoder), mMarco (Cross-Encoder) and RapidFuzz
"""
Hybrid Entity Resolution Engine (v1.0)
--------------------------------------
Author: Válber Azevedo
Date: 2026-02-13

Description:
    A multi-stage retrieval pipeline designed for high-stakes financial reconciliation.
    It combines three distinct layers of analysis to resolve entity identity:
    
    1. Cross-Encoder (unicamp-dl/mMiniLM-L6-v2-mmarco-v2): 
       Acts as the "Judge". High precision for B2B/Corporate entities.
       
    2. Bi-Encoder (BAAI/bge-m3): 
       Acts as the "Context Lawyer". Captures semantic relationships (e.g., Zé = José).
       
    3. Fuzzy Match (RapidFuzz): 
       Acts as the "Scribe". Handles raw typos and syntactic errors.

Usage:
    resolver = EntityResolver()
    results = resolver.batch_resolve(queries, targets)
"""

