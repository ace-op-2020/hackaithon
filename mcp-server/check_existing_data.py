#!/usr/bin/env python3
"""
Quick check of existing data in trellix_document_embeddings table
"""

import asyncio
from google.cloud import spanner

async def check_existing_data():
    """Check what data exists in the trellix_document_embeddings table"""
    
    try:
        print("🔍 Checking existing data in trellix_document_embeddings...")
        
        # Initialize Spanner client
        spanner_client = spanner.Client()
        instance = spanner_client.instance("trellix-knowledge-graph")
        database = instance.database("knowledge_graph_db")
        
        with database.snapshot() as snapshot:
            # Check table structure and sample data
            results = snapshot.execute_sql("""
                SELECT doc_id, title, source_type, 
                       LENGTH(content) as content_length,
                       ARRAY_LENGTH(embedding) as embedding_dim
                FROM trellix_document_embeddings 
                LIMIT 10
            """)
            
            print("📊 Sample data from trellix_document_embeddings:")
            print("-" * 80)
            
            count = 0
            for row in results:
                count += 1
                doc_id = row[0]
                title = row[1] or "Untitled"
                source_type = row[2] or "unknown"
                content_len = row[3] or 0
                embedding_dim = row[4] or 0
                
                print(f"[{count}] ID: {doc_id[:20]}...")
                print(f"    Title: {title}")
                print(f"    Source: {source_type}")
                print(f"    Content Length: {content_len} chars")
                print(f"    Embedding Dimension: {embedding_dim}")
                print()
            
            if count == 0:
                print("⚠️ No data found in trellix_document_embeddings table")
            else:
                # Get total count
                count_results = snapshot.execute_sql("""
                    SELECT COUNT(*) FROM trellix_document_embeddings
                """)
                total_count = list(count_results)[0][0]
                print(f"✅ Total documents in table: {total_count}")
                
    except Exception as e:
        print(f"❌ Error checking data: {e}")
        print("Make sure the table exists and has the correct structure")

if __name__ == "__main__":
    asyncio.run(check_existing_data())