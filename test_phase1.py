"""
Test Phase 1 Enhancements
Quick test script to verify all Phase 1 features are working.
"""

from rag_engine import get_rag_engine
import json


def test_phase1_features():
    """Test all Phase 1 enhancements."""
    
    print("=" * 70)
    print("PHASE 1 ENHANCEMENTS TEST")
    print("=" * 70)
    
    # Initialize engine
    print("\n1. Initializing RAG Engine...")
    engine = get_rag_engine()
    
    if not engine.is_initialized:
        print("❌ Engine failed to initialize")
        return
    
    print("✓ Engine initialized successfully")
    
    # Check components
    print("\n2. Checking Phase 1 Components...")
    print(f"   Reranker: {'✓ Enabled' if engine.reranker else '✗ Disabled'}")
    print(f"   Validator: {'✓ Enabled' if engine.validator else '✗ Disabled'}")
    print(f"   Evaluator: {'✓ Enabled' if engine.evaluator else '✗ Disabled'}")
    print(f"   Cache: {'✓ Enabled' if engine.cache else '✗ Disabled'}")
    
    # Test query
    print("\n3. Testing Query with Phase 1 Pipeline...")
    test_question = "What are the Miranda rights requirements in criminal procedure?"
    
    print(f"   Question: {test_question}")
    result = engine.query(test_question)
    
    if "error" in result:
        print(f"❌ Query failed: {result['error']}")
        return
    
    print("✓ Query completed successfully")
    
    # Display results
    print("\n4. Response Details:")
    print(f"   Answer length: {len(result.get('answer', ''))} characters")
    print(f"   Sources: {result.get('num_sources', 0)}")
    print(f"   Citations: {len(result.get('citations', []))}")
    
    # Cache info
    if result.get("cache_hit"):
        print(f"\n   🔥 Cache Hit! Type: {result.get('cache_type', 'unknown')}")
        if "similarity_score" in result:
            print(f"      Similarity: {result['similarity_score']}")
    
    # Reranker info
    if "reranker_metadata" in result:
        meta = result["reranker_metadata"]
        print(f"\n   📊 Reranker Stats:")
        print(f"      Documents reranked: {meta.get('num_documents', 0)}")
        print(f"      Order changed: {meta.get('order_changed', False)}")
        if "scores" in meta:
            print(f"      Score mean: {meta['scores'].get('mean', 0):.4f}")
            print(f"      Score max: {meta['scores'].get('max', 0):.4f}")
    
    # Validation info
    if "validation" in result:
        val = result["validation"]
        print(f"\n   ✅ Validation Results:")
        print(f"      Passed: {val.get('passed', False)}")
        print(f"      Confidence: {val.get('confidence', 0):.2f}")
        print(f"      Recommendation: {val.get('recommendation', 'N/A')}")
    
    # Evaluation info
    if "evaluation" in result:
        eval_data = result["evaluation"]
        print(f"\n   📈 Evaluation Metrics:")
        if "overall_score" in eval_data:
            print(f"      Overall Score: {eval_data['overall_score']}/5")
            print(f"      Quality: {eval_data.get('overall_quality', 'N/A')}")
        
        if "faithfulness" in eval_data:
            print(f"      Faithfulness: {eval_data['faithfulness'].get('score', 0)}/5")
        if "relevancy" in eval_data:
            print(f"      Relevancy: {eval_data['relevancy'].get('score', 0)}/5")
    
    # Usage stats
    if "usage" in result:
        usage = result["usage"]
        print(f"\n   💰 Token Usage:")
        print(f"      Total tokens: {usage.get('total_tokens', 0)}")
        print(f"      Cost: ${usage.get('total_cost', 0):.4f}")
    
    # Test second query (should hit cache)
    print("\n5. Testing Cache with Same Query...")
    result2 = engine.query(test_question)
    
    if result2.get("cache_hit"):
        print("✓ Cache working! Second query returned cached result")
    else:
        print("⚠ Cache miss on identical query (might need adjustment)")
    
    # Get overall statistics
    print("\n6. Overall System Statistics:")
    stats = engine.get_stats()
    
    if "cache" in stats:
        cache_stats = stats["cache"]
        print(f"   Cache:")
        print(f"      Hit rate: {cache_stats.get('hit_rate', 0):.2%}")
        print(f"      Total requests: {cache_stats.get('total_requests', 0)}")
        print(f"      Cache size: {cache_stats.get('cache_size', 0)}/{cache_stats.get('max_size', 0)}")
    
    if "evaluation" in stats:
        eval_stats = stats["evaluation"]
        print(f"   Evaluation:")
        print(f"      Evaluations run: {eval_stats.get('num_evaluations', 0)}")
        if "average_generation_score" in eval_stats:
            print(f"      Average score: {eval_stats['average_generation_score']:.2f}/5")
    
    # Print sample answer
    print("\n7. Sample Answer (first 500 chars):")
    print("-" * 70)
    answer = result.get("answer", "")
    print(answer[:500] + ("..." if len(answer) > 500 else ""))
    print("-" * 70)
    
    print("\n" + "=" * 70)
    print("✅ PHASE 1 TEST COMPLETED SUCCESSFULLY")
    print("=" * 70)
    
    # Save detailed results
    print("\n💾 Saving detailed results to phase1_test_results.json...")
    test_results = {
        "test_question": test_question,
        "components_enabled": {
            "reranker": engine.reranker is not None,
            "validator": engine.validator is not None,
            "evaluator": engine.evaluator is not None,
            "cache": engine.cache is not None
        },
        "first_query_result": {
            "answer_length": len(result.get("answer", "")),
            "num_sources": result.get("num_sources", 0),
            "cache_hit": result.get("cache_hit", False),
            "validation": result.get("validation", {}),
            "evaluation": result.get("evaluation", {}),
            "usage": result.get("usage", {})
        },
        "second_query_result": {
            "cache_hit": result2.get("cache_hit", False)
        },
        "overall_stats": stats
    }
    
    with open("phase1_test_results.json", "w") as f:
        json.dump(test_results, f, indent=2)
    
    print("✓ Results saved to phase1_test_results.json")
    
    print("\n📚 Next Steps:")
    print("   1. Review PHASE1_GUIDE.md for detailed documentation")
    print("   2. Check phase1_test_results.json for complete test data")
    print("   3. Check evaluation_history.json for evaluation details")
    print("   4. Adjust config.py settings as needed")
    print("   5. Run your own queries to test the system")


if __name__ == "__main__":
    try:
        test_phase1_features()
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
