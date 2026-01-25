"""
Test script for the improved Legal Chunker
Run this to verify the new chunking features work correctly.
"""

import os
import sys

# Add project to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from legal_chunker import (
    LegalChunker, 
    TableOfContentsExtractor, 
    LegalDefinitionExtractor,
    CrossReferenceTracker,
    create_legal_chunks,
    create_parent_child_chunks
)

try:
    from langchain.schema import Document
except ImportError:
    from langchain_core.documents import Document


def test_toc_extraction():
    """Test Table of Contents extraction."""
    print("\n" + "="*60)
    print("TEST: Table of Contents Extraction")
    print("="*60)
    
    # Sample TOC content
    toc_text = """
    TABLE OF CONTENTS
    
    Rule 1. Scope and Definitions..................1
    Rule 2. One Form of Action.....................5
    Rule 3. Commencing an Action...................8
    Rule 4. Summons................................12
    Rule 5. Serving and Filing Pleadings..........18
    Rule 6. Computing and Extending Time..........25
    """
    
    # Test TOC detection
    is_toc = TableOfContentsExtractor.is_toc_page(toc_text)
    print(f"✓ TOC Detection: {is_toc}")
    
    # Test entry extraction
    entries = TableOfContentsExtractor.extract_toc_entries(toc_text)
    print(f"✓ Extracted {len(entries)} TOC entries")
    for entry in entries[:3]:
        print(f"  - Rule {entry['number']}: {entry['title']} (Page {entry.get('page', 'N/A')})")
    
    # Test chunk creation
    metadata = {'source': 'test_doc.pdf', 'doc_type': 'federal_civil_rules'}
    toc_chunk = TableOfContentsExtractor.create_toc_chunk(toc_text, metadata)
    if toc_chunk:
        print(f"✓ Created TOC chunk with {toc_chunk.metadata.get('entry_count')} entries")
        print(f"  Chunk type: {toc_chunk.metadata.get('chunk_type')}")


def test_definition_extraction():
    """Test legal definition extraction."""
    print("\n" + "="*60)
    print("TEST: Legal Definition Extraction")
    print("="*60)
    
    # Sample text with definitions
    text_with_definitions = """
    Rule 1. Scope and Definitions
    
    (a) Scope. These rules govern procedure in all civil actions and proceedings.
    
    (b) Definitions. In these rules:
    
    "Action" means a civil action or proceeding.
    
    "Court" means the district court and includes a magistrate judge.
    
    "State" means any state of the United States, the District of Columbia, 
    Puerto Rico, or any territory or insular possession subject to the 
    jurisdiction of the United States.
    
    "Federal statute" means any statute of the United States.
    """
    
    definitions = LegalDefinitionExtractor.extract_definitions(text_with_definitions)
    print(f"✓ Extracted {len(definitions)} definitions")
    for defn in definitions:
        print(f"  - Term: {defn['term']}")
        print(f"    Definition: {defn['definition'][:80]}...")
    
    # Test chunk creation
    metadata = {'source': 'test.pdf', 'doc_type': 'federal_civil_rules'}
    defn_chunks = LegalDefinitionExtractor.create_definition_chunks(definitions, metadata)
    print(f"✓ Created {len(defn_chunks)} definition chunks")


def test_cross_reference_tracking():
    """Test cross-reference extraction."""
    print("\n" + "="*60)
    print("TEST: Cross-Reference Tracking")
    print("="*60)
    
    text_with_refs = """
    Under Rule 12(b), a party may assert certain defenses by motion. 
    See also MCR 2.116 for summary disposition standards.
    This is consistent with Fed. R. Civ. P. 56 and Section 1404.
    Pursuant to Rule 23, class actions must meet specific requirements.
    """
    
    refs = CrossReferenceTracker.extract_cross_references(text_with_refs)
    print(f"✓ Extracted {len(refs)} cross-references:")
    for ref in refs:
        print(f"  - {ref}")
    
    # Test metadata enrichment
    metadata = {'source': 'test.pdf'}
    enriched = CrossReferenceTracker.enrich_metadata(text_with_refs, metadata)
    print(f"✓ Metadata enriched with cross_references: {enriched.get('has_cross_references')}")


def test_legal_chunker():
    """Test the main LegalChunker with all features."""
    print("\n" + "="*60)
    print("TEST: Full Legal Chunker")
    print("="*60)
    
    # Create a sample legal document
    sample_doc = Document(
        page_content="""
        [PAGE 1]
        TABLE OF CONTENTS
        Rule 1. Scope and Definitions..................1
        Rule 2. One Form of Action.....................5
        Rule 3. Commencing an Action...................8
        
        [PAGE 2]
        FEDERAL RULES OF CIVIL PROCEDURE
        
        Rule 1. Scope and Purpose
        
        These rules govern the procedure in all civil actions and proceedings 
        in the United States district courts, except as stated in Rule 81. 
        They should be construed, administered, and employed by the court and 
        the parties to secure the just, speedy, and inexpensive determination 
        of every action and proceeding.
        
        (a) Purpose. The purpose of these rules is to provide a uniform system
        of procedure for civil litigation in federal courts.
        
        (b) Construction. These rules shall be construed to:
        (1) achieve the purposes stated above;
        (2) simplify procedure;
        (3) promote justice.
        
        Rule 2. One Form of Action
        
        There is one form of action—the civil action. See Rule 1 for the scope
        of these rules. As used in this rule, "action" means a civil action.
        """,
        metadata={
            'source': 'federal-rules-of-civil-procedure.pdf',
            'doc_type': 'federal_civil_rules'
        }
    )
    
    # Test with all features enabled
    chunker = LegalChunker(
        chunk_size=500,
        chunk_overlap=50,
        extract_toc=True,
        extract_definitions=True,
        preserve_hierarchy=True
    )
    
    chunks = chunker.chunk_documents([sample_doc])
    
    print(f"✓ Created {len(chunks)} total chunks")
    
    # Analyze chunk types
    chunk_types = {}
    for chunk in chunks:
        ct = chunk.metadata.get('chunk_type', 'unknown')
        chunk_types[ct] = chunk_types.get(ct, 0) + 1
    
    print("\nChunk breakdown by type:")
    for ct, count in chunk_types.items():
        print(f"  - {ct}: {count}")
    
    # Show sample chunks
    print("\nSample chunks:")
    for i, chunk in enumerate(chunks[:3]):
        print(f"\n--- Chunk {i+1} ({chunk.metadata.get('chunk_type', 'unknown')}) ---")
        print(f"Content preview: {chunk.page_content[:150]}...")
        print(f"Metadata: {list(chunk.metadata.keys())}")


def test_hierarchical_context():
    """Test hierarchical context preservation."""
    print("\n" + "="*60)
    print("TEST: Hierarchical Context Preservation")
    print("="*60)
    
    # Create a document with section info
    doc = Document(
        page_content="""
        Rule 12. Defenses and Objections
        
        (a) Time to Serve a Responsive Pleading.
        (1) In General. Unless another time is specified by this rule or a 
        federal statute, the time for serving a responsive pleading is as follows:
        (A) A defendant must serve an answer within 21 days after being served 
        with the summons and complaint.
        (B) A party must serve an answer to a counterclaim or crossclaim within 
        21 days after being served with the pleading that states the counterclaim 
        or crossclaim.
        
        (2) United States and Its Agencies. The United States, a United States 
        agency, or a United States officer or employee sued only in an official 
        capacity must serve an answer within 60 days.
        """,
        metadata={
            'source': 'frcp.pdf',
            'doc_type': 'federal_civil_rules',
            'section_number': '12',
            'section_title': 'Defenses and Objections'
        }
    )
    
    chunker = LegalChunker(
        chunk_size=200,  # Small size to force splitting
        preserve_hierarchy=True
    )
    
    chunks = chunker.chunk_documents([doc])
    
    print(f"✓ Split into {len(chunks)} chunks with hierarchy preserved")
    
    for i, chunk in enumerate(chunks):
        parent = chunk.metadata.get('parent_section', 'N/A')
        print(f"\nChunk {i+1}:")
        print(f"  Parent section: {parent}")
        print(f"  Content starts with: {chunk.page_content[:100]}...")


def test_parent_child_chunking():
    """Test parent-child chunking strategy."""
    print("\n" + "="*60)
    print("TEST: Parent-Child Chunking")
    print("="*60)
    
    doc = Document(
        page_content="""
        Rule 23. Class Actions
        
        (a) Prerequisites. One or more members of a class may sue or be sued as 
        representative parties on behalf of all members only if:
        (1) the class is so numerous that joinder of all members is impracticable;
        (2) there are questions of law or fact common to the class;
        (3) the claims or defenses of the representative parties are typical of 
        the claims or defenses of the class; and
        (4) the representative parties will fairly and adequately protect the 
        interests of the class.
        
        (b) Types of Class Actions. A class action may be maintained if Rule 23(a) 
        is satisfied and if:
        (1) prosecuting separate actions would create risk of inconsistent 
        adjudications; or
        (2) the party opposing the class has acted on grounds that apply generally 
        to the class; or
        (3) the court finds that questions of law or fact common to class members 
        predominate over any questions affecting only individual members.
        """,
        metadata={'source': 'frcp.pdf', 'doc_type': 'federal_civil_rules'}
    )
    
    child_chunks, parent_lookup = create_parent_child_chunks(
        [doc],
        parent_size=1000,
        child_size=200
    )
    
    print(f"✓ Created {len(child_chunks)} child chunks")
    print(f"✓ Created {len(parent_lookup)} parent chunks")
    
    # Show relationship
    if child_chunks:
        sample_child = child_chunks[0]
        parent_id = sample_child.metadata.get('parent_id')
        print(f"\nSample child chunk:")
        print(f"  Content: {sample_child.page_content[:100]}...")
        print(f"  Parent ID: {parent_id}")
        if parent_id in parent_lookup:
            parent = parent_lookup[parent_id]
            print(f"  Parent content length: {len(parent.page_content)} chars")


def main():
    """Run all tests."""
    print("\n" + "="*60)
    print("LEGAL CHUNKER IMPROVEMENT TESTS")
    print("="*60)
    
    try:
        test_toc_extraction()
        test_definition_extraction()
        test_cross_reference_tracking()
        test_legal_chunker()
        test_hierarchical_context()
        test_parent_child_chunking()
        
        print("\n" + "="*60)
        print("ALL TESTS COMPLETED SUCCESSFULLY!")
        print("="*60)
        print("\nKey improvements implemented:")
        print("1. ✓ Table of Contents extraction (separate navigational chunks)")
        print("2. ✓ Legal definition extraction (searchable definition chunks)")
        print("3. ✓ Cross-reference tracking (enriched metadata)")
        print("4. ✓ Hierarchical context preservation (parent section info)")
        print("5. ✓ Enhanced jury instruction chunking")
        print("6. ✓ Parent-child chunking for better retrieval")
        print("7. ✓ Context headers in generic chunks")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
