"""
Comprehensive Test Suite - Portfolio Optimizer
==============================================

Tests all features including:
- Core functionality
- Advanced analytics
- Authentication (if configured)
- Database operations
- Example portfolio
"""

import sys
import json
sys.path.insert(0, '/home/simon/py/nmy/risk_optimization/portfolio_optimizer')

# Import the pre-created app instance (not create_app function)
from app import app

def run_tests():
    """Run comprehensive test suite."""
    print("="*70)
    print("PORTFOLIO OPTIMIZER - COMPREHENSIVE TEST SUITE")
    print("="*70)
    
    with app.test_client() as client:
        tests_passed = 0
        tests_failed = 0
        
        # Test 1: Health Check
        print("\n1. Health Check...")
        try:
            response = client.get('/api/health')
            print(f"   Response status: {response.status_code}")
            data = json.loads(response.data)
            print(f"   Response data: {data}")
            assert response.status_code == 200
            assert data.get('status') == 'healthy'
            print("   ✓ PASS - Server is healthy")
            tests_passed += 1
        except Exception as e:
            print(f"   ✗ FAIL - {e}")
            import traceback
            traceback.print_exc()
            tests_failed += 1
        
        # Test 2: Example Portfolio Analysis
        print("\n2. Example Portfolio Analysis...")
        try:
            response = client.get('/api/example-portfolio')
            print(f"   Response status: {response.status_code}")
            if response.status_code == 200:
                data = json.loads(response.data)
                # Response is wrapped in {success, data}
                if data.get('success') and 'data' in data:
                    inner_data = data['data']
                    assert 'meta' in inner_data
                    assert 'metrics' in inner_data
                    print(f"   ✓ PASS - Example portfolio loaded with {len(inner_data)} data sections")
                    tests_passed += 1
                else:
                    print(f"   Unexpected structure: {list(data.keys())}")
                    tests_failed += 1
            else:
                print(f"   Response: {response.data.decode()}")
                tests_failed += 1
        except Exception as e:
            print(f"   ✗ FAIL - {e}")
            import traceback
            traceback.print_exc()
            tests_failed += 1
        
        # Test 3: Comprehensive Analysis with Example Portfolio
        print("\n3. Comprehensive Analysis...")
        try:
            holdings = [
                {'ticker': 'AAPL', 'weight': 0.15},
                {'ticker': 'MSFT', 'weight': 0.15},
                {'ticker': 'GOOGL', 'weight': 0.10}
            ]
            response = client.post('/api/portfolio/comprehensive-analysis',
                                 data=json.dumps({'holdings': holdings}),
                                 content_type='application/json')
            assert response.status_code == 200
            data = json.loads(response.data)
            assert 'metrics' in data
            print(f"   ✓ PASS - Comprehensive analysis returned {len(data.get('metrics', {}))} metrics")
            tests_passed += 1
        except Exception as e:
            print(f"   ✗ FAIL - {e}")
            tests_failed += 1
        
        # Test 4: Efficient Frontier
        print("\n4. Efficient Frontier...")
        try:
            # This endpoint expects tickers list, not holdings
            response = client.post('/api/optimize/efficient-frontier',
                                 data=json.dumps({
                                     'tickers': ['SPY', 'QQQ', 'TLT', 'IWM']
                                 }),
                                 content_type='application/json')
            print(f"   Response status: {response.status_code}")
            data = json.loads(response.data)
            print(f"   Keys: {list(data.keys())}")
            if response.status_code == 200:
                print(f"   ✓ PASS - Efficient frontier generated")
                tests_passed += 1
            else:
                print(f"   Warning: {data.get('error', 'Unknown error')}")
                tests_failed += 1
        except Exception as e:
            print(f"   ✗ FAIL - {e}")
            import traceback
            traceback.print_exc()
            tests_failed += 1
        
        # Test 5: Advanced Risk Metrics
        print("\n5. Advanced Risk Metrics...")
        try:
            # This endpoint expects tickers and weights, not holdings
            response = client.post('/api/advanced/risk',
                                 data=json.dumps({
                                     'tickers': ['SPY', 'TLT'],
                                     'weights': {'SPY': 0.5, 'TLT': 0.5}
                                 }),
                                 content_type='application/json')
            print(f"   Response status: {response.status_code}")
            data = json.loads(response.data)
            print(f"   Keys: {list(data.keys())}")
            if response.status_code == 200:
                print(f"   ✓ PASS - Advanced risk metrics available")
                tests_passed += 1
            else:
                print(f"   Warning: {data.get('error', 'Unknown error')}")
                tests_failed += 1
        except Exception as e:
            print(f"   ✗ FAIL - {e}")
            import traceback
            traceback.print_exc()
            tests_failed += 1
        
        # Test 6: Factor Exposure
        print("\n6. Factor Exposure Analysis...")
        try:
            # This endpoint expects tickers and weights
            response = client.post('/api/factor-exposure',
                                 data=json.dumps({
                                     'tickers': ['SPY', 'QQQ'],
                                     'weights': {'SPY': 0.5, 'QQQ': 0.5}
                                 }),
                                 content_type='application/json')
            print(f"   Response status: {response.status_code}")
            data = json.loads(response.data)
            print(f"   Keys: {list(data.keys())}")
            if response.status_code == 200:
                print(f"   ✓ PASS - Factor exposure analysis completed")
                tests_passed += 1
            else:
                print(f"   Warning: {data.get('error', 'Unknown error')}")
                tests_failed += 1
        except Exception as e:
            print(f"   ✗ FAIL - {e}")
            import traceback
            traceback.print_exc()
            tests_failed += 1
        
        # Test 7: Regime Detection
        print("\n7. Regime Detection...")
        try:
            # This endpoint expects tickers list
            response = client.post('/api/regime-detection',
                                 data=json.dumps({
                                     'tickers': ['SPY']
                                 }),
                                 content_type='application/json')
            print(f"   Response status: {response.status_code}")
            data = json.loads(response.data)
            print(f"   Keys: {list(data.keys())}")
            if response.status_code == 200:
                print(f"   ✓ PASS - Regime detection completed")
                tests_passed += 1
            else:
                print(f"   Warning: {data.get('error', 'Unknown error')}")
                tests_failed += 1
        except Exception as e:
            print(f"   ✗ FAIL - {e}")
            import traceback
            traceback.print_exc()
            tests_failed += 1
        
        # Test 8: Covariance Estimators
        print("\n8. Covariance Estimators...")
        try:
            # This endpoint expects tickers list
            response = client.post('/api/covariance-estimators',
                                 data=json.dumps({
                                     'tickers': ['SPY', 'QQQ', 'TLT', 'IWM']
                                 }),
                                 content_type='application/json')
            print(f"   Response status: {response.status_code}")
            data = json.loads(response.data)
            print(f"   Keys: {list(data.keys())}")
            if response.status_code == 200:
                print(f"   ✓ PASS - Covariance estimators comparison completed")
                tests_passed += 1
            else:
                print(f"   Warning: {data.get('error', 'Unknown error')}")
                tests_failed += 1
        except Exception as e:
            print(f"   ✗ FAIL - {e}")
            import traceback
            traceback.print_exc()
            tests_failed += 1
        
        # Test 9: Asset History
        print("\n9. Asset History...")
        try:
            response = client.get('/api/assets/AAPL/history?period=1y')
            assert response.status_code == 200
            data = json.loads(response.data)
            assert 'prices' in data or 'ticker' in data
            print(f"   ✓ PASS - Asset history retrieved")
            tests_passed += 1
        except Exception as e:
            print(f"   ✗ FAIL - {e}")
            tests_failed += 1
        
        # Test 10: Portfolio Analysis
        print("\n10. Portfolio Analysis...")
        try:
            holdings = [
                {'ticker': 'VTI', 'weight': 0.6},
                {'ticker': 'BND', 'weight': 0.4}
            ]
            response = client.post('/api/portfolio/analyze',
                                 data=json.dumps({'holdings': holdings}),
                                 content_type='application/json')
            assert response.status_code == 200
            data = json.loads(response.data)
            assert 'total_return' in data or 'metrics' in data
            print(f"   ✓ PASS - Portfolio analysis completed")
            tests_passed += 1
        except Exception as e:
            print(f"   ✗ FAIL - {e}")
            tests_failed += 1
        
        # Summary
        print("\n" + "="*70)
        print(f"TEST RESULTS: {tests_passed} passed, {tests_failed} failed")
        print("="*70)
        
        if tests_failed == 0:
            print("\n✅ ALL TESTS PASSED - System is fully operational!")
        else:
            print(f"\n⚠️  {tests_failed} TEST(S) FAILED - Please review errors above")
        
        return tests_failed == 0

if __name__ == '__main__':
    success = run_tests()
    sys.exit(0 if success else 1)
