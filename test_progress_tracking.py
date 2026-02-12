"""
Test script to verify progress tracking implementation.

This script tests the progress logging functionality by running a simple query
and checking that progress events are being written to the log file.
"""
import json
import os
import time
from pathlib import Path

import carnot


def test_progress_tracking():
    """Test that progress events are logged correctly."""
    print("Testing progress tracking implementation...")
    
    # Create a test session directory
    session_id = "test-session-123"
    sessions_dir = Path("app/backend/sessions")
    sessions_dir.mkdir(exist_ok=True)
    
    session_dir = sessions_dir / session_id
    session_dir.mkdir(exist_ok=True)
    
    progress_log = session_dir / "progress.jsonl"
    
    # Remove old log if exists
    if progress_log.exists():
        progress_log.unlink()
    
    print(f"Session directory: {session_dir}")
    print(f"Progress log: {progress_log}")
    
    # Create a simple context and run a query
    ctx = carnot.TextFileContext(
        "data/enron-eval-medium",
        "test-context",
        "Test emails for progress tracking"
    )
    
    # Run a simple query with progress tracking enabled
    config = carnot.QueryProcessorConfig(
        policy=carnot.MaxQuality(),
        progress=False,  # Disable console progress
        session_id=session_id,
        progress_log_file=str(progress_log),
        num_samples=5,  # Only process 5 samples for testing
    )
    
    print("\nRunning query with progress tracking...")
    start_time = time.time()
    
    try:
        output = ctx.run(config=config)
        elapsed = time.time() - start_time
        
        print(f"\nQuery completed in {elapsed:.2f}s")
        print(f"Results: {len(output.data_records)} records")
        
        # Check if progress log was created
        if not progress_log.exists():
            print("❌ ERROR: Progress log file was not created!")
            return False
        
        # Read and analyze progress events
        events = []
        with open(progress_log, 'r') as f:
            for line in f:
                if line.strip():
                    events.append(json.loads(line))
        
        print(f"\n✅ Progress log created with {len(events)} events")
        
        # Analyze events
        operators = {}
        for event in events:
            op_id = event['operator_id']
            if op_id not in operators:
                operators[op_id] = {
                    'name': event['operator_name'],
                    'events': [],
                    'level': event['level']
                }
            operators[op_id]['events'].append(event)
        
        print(f"\nTracked {len(operators)} operator(s):")
        for op_id, op_data in operators.items():
            level = op_data['level']
            name = op_data['name']
            num_events = len(op_data['events'])
            last_event = op_data['events'][-1]
            status = last_event['status']
            percentage = last_event['progress']['percentage']
            cost = last_event['cost']
            
            indent = "  " if level == "inner" else ""
            print(f"{indent}• {name} ({level}): {num_events} events, {status}, {percentage:.1f}%, ${cost:.4f}")
        
        # Verify we have at least one completed operator
        completed = sum(1 for op in operators.values() 
                       if any(e['status'] == 'completed' for e in op['events']))
        
        if completed > 0:
            print(f"\n✅ Test PASSED: {completed} operator(s) completed successfully")
            return True
        else:
            print("\n❌ Test FAILED: No operators completed")
            return False
            
    except Exception as e:
        print(f"\n❌ Error during test: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_progress_tracking()
    exit(0 if success else 1)


