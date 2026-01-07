// Test Script for Custom Bytecode VM
// Calculates 10 + 20 and prints results

PUSH_STR "Starting Test Script"
API_CALL LOG

// Test Arithmetic
PUSH_FLOAT 10.0
PUSH_FLOAT 20.0
ADD
PUSH_STR "10 + 20 =" 
PRINT
PRINT 

// Test Logic
PUSH_INT 5
PUSH_INT 10
LT 
PUSH_STR "5 < 10 is:"
PRINT
PRINT

// Test API Call
PUSH_FLOAT 1.0
PUSH_FLOAT 2.0
PUSH_FLOAT 3.0
API_CALL VEC3_CREATE

PUSH_STR "Script Finished"
API_CALL LOG
HALT
