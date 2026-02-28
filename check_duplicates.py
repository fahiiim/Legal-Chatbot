"""
Check for duplicate chunks in ChromaDB
"""
import chromadb
import hashlib

print("Connecting to ChromaDB...")
chroma_client = chromadb.PersistentClient(path="./chroma_db")

collections = chroma_client.list_collections()
print(f"Found {len(collections)} collection(s)\n")

total_chunks = 0
total_duplicates = 0

for collection in collections:
    col = chroma_client.get_collection(collection.name)
    count = col.count()
    print(f"Collection : {collection.name}")
    print(f"Total chunks: {count:,}")

    # First get all IDs only (no content, avoids memory crash)
    print("Fetching all IDs...")
    all_ids_result = col.get(include=[])
    all_ids = all_ids_result["ids"]
    print(f"Retrieved {len(all_ids):,} IDs")

    # Fetch content in small batches using IDs
    batch_size = 500
    hash_map = {}
    duplicate_ids = []

    for i in range(0, len(all_ids), batch_size):
        batch_ids = all_ids[i:i + batch_size]
        try:
            results = col.get(ids=batch_ids, include=["documents"])
            for id_, text in zip(results["ids"], results["documents"]):
                if text is None:
                    continue
                text_hash = hashlib.md5(text.encode()).hexdigest()
                if text_hash in hash_map:
                    duplicate_ids.append(id_)
                else:
                    hash_map[text_hash] = id_
        except Exception as e:
            print(f"  Warning: batch {i}-{i+batch_size} failed: {e}")
            continue

        done = min(i + batch_size, len(all_ids))
        print(f"  Processed {done:,}/{len(all_ids):,}...", end="\r")

    print()
    total_chunks += count
    total_duplicates += len(duplicate_ids)
    unique = len(hash_map)
    dupes = len(duplicate_ids)

    print(f"  Unique chunks    : {unique:,}")
    print(f"  Duplicate chunks : {dupes:,}")
    if count > 0:
        print(f"  Duplicate %%     : {(dupes/count)*100:.1f}%%")

print(f"\n{'='*55}")
print(f"TOTAL CHUNKS              : {total_chunks:,}")
print(f"TOTAL DUPLICATES          : {total_duplicates:,}")
print(f"UNIQUE CHUNKS             : {total_chunks - total_duplicates:,}")
print(f"WASTED SPACE (estimated)  : ~{(total_duplicates * 1.5) / 1024:.2f} GB")
print(f"{'='*55}")
