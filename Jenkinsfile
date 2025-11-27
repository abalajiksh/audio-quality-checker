pipeline {
    agent any
    
    // Build parameters - allows manual trigger with options
    parameters {
        choice(
            name: 'TEST_TYPE_OVERRIDE',
            choices: ['AUTO', 'QUALIFICATION', 'REGRESSION'],
            description: 'Force a specific test type. AUTO uses smart detection.'
        )
        booleanParam(
            name: 'SKIP_SONARQUBE',
            defaultValue: false,
            description: 'Skip SonarQube analysis'
        )
        booleanParam(
            name: 'CLEAN_WORKSPACE_BEFORE',
            defaultValue: false,
            description: 'Clean workspace before build (use if seeing stale file issues)'
        )
    }
    
    environment {
        // MinIO configuration
        MINIO_BUCKET = 'audiocheckr'
        MINIO_FILE_COMPACT = 'CompactTestFiles.zip'
        MINIO_FILE_FULL = 'TestFiles.zip'
        
        // SonarQube configuration
        SONAR_PROJECT_KEY = 'audiocheckr'
        SONAR_PROJECT_NAME = 'AudioCheckr'
        SONAR_SOURCES = 'src'
        
        // Path setup
        PATH = "$HOME/bin:$HOME/.cargo/bin:/usr/bin:$PATH"
    }
    
    triggers {
        // Scheduled regression test - Saturday at 10:00 AM
        cron('0 10 * * 6')
    }
    
    options {
        // Build timeout (prevent stuck builds)
        timeout(time: 60, unit: 'MINUTES')
        
        // Keep last 10 builds
        buildDiscarder(logRotator(numToKeepStr: '10', artifactNumToKeepStr: '5'))
        
        // Add timestamps to console output
        timestamps()
        
        // Don't run concurrent builds
        disableConcurrentBuilds()
    }
    
    stages {
        stage('Pre-flight') {
            steps {
                script {
                    // Clean workspace if requested
                    if (params.CLEAN_WORKSPACE_BEFORE) {
                        deleteDir()
                        checkout scm
                    }
                    
                    // Determine test type
                    if (params.TEST_TYPE_OVERRIDE && params.TEST_TYPE_OVERRIDE != 'AUTO') {
                        env.TEST_TYPE = params.TEST_TYPE_OVERRIDE
                        echo "🔧 Test type forced via parameter: ${env.TEST_TYPE}"
                    } else if (currentBuild.getBuildCauses('hudson.triggers.TimerTrigger$TimerTriggerCause')) {
                        // Scheduled build (cron) = REGRESSION
                        env.TEST_TYPE = 'REGRESSION'
                        echo "⏰ Scheduled build detected - running REGRESSION tests"
                    } else if (currentBuild.getBuildCauses('hudson.model.Cause$UserIdCause')) {
                        // Manual build = QUALIFICATION by default
                        env.TEST_TYPE = 'QUALIFICATION'
                        echo "👤 Manual build - running QUALIFICATION tests (use parameter to override)"
                    } else {
                        // GitHub push = QUALIFICATION
                        env.TEST_TYPE = 'QUALIFICATION'
                        echo "🔄 Push detected - running QUALIFICATION tests"
                    }
                    
                    // Display build info
                    echo """
========================================================
                  AUDIOCHECKR CI/CD                     
========================================================
  Test Type:     ${env.TEST_TYPE}
  Build #:       ${currentBuild.number}
  Triggered by:  ${currentBuild.getBuildCauses()[0].shortDescription}
========================================================
"""
                }
            }
        }
        
        stage('Setup Tools') {
            steps {
                sh '''
                    mkdir -p $HOME/bin
                    
                    if ! command -v cc >/dev/null 2>&1; then
                        echo "ERROR: C compiler not found!"
                        exit 1
                    fi
                    
                    if ! command -v mc >/dev/null 2>&1; then
                        echo "Installing MinIO client..."
                        wget -q https://dl.min.io/client/mc/release/linux-amd64/mc -O $HOME/bin/mc
                        chmod +x $HOME/bin/mc
                    fi
                    
                    if ! command -v cargo >/dev/null 2>&1; then
                        echo "Installing Rust..."
                        curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
                        . $HOME/.cargo/env
                    fi
                    
                    echo "=== Tool Versions ==="
                    mc --version
                    cargo --version
                    rustc --version
                    echo "===================="
                '''
            }
        }
        
        stage('Checkout') {
            steps {
                checkout scm
                script {
                    env.GIT_COMMIT_MSG = sh(script: 'git log -1 --pretty=%B', returnStdout: true).trim()
                    env.GIT_COMMIT_SHORT = sh(script: 'git rev-parse --short HEAD', returnStdout: true).trim()
                    env.GIT_AUTHOR = sh(script: 'git log -1 --pretty=%an', returnStdout: true).trim()
                    
                    echo "Commit: ${env.GIT_COMMIT_SHORT} by ${env.GIT_AUTHOR}"
                    echo "Message: ${env.GIT_COMMIT_MSG}"
                }
            }
        }
        
        stage('Download Test Files') {
            steps {
                script {
                    withCredentials([
                        usernamePassword(
                            credentialsId: 'noIdea',
                            usernameVariable: 'MINIO_ACCESS_KEY',
                            passwordVariable: 'MINIO_SECRET_KEY'
                        ),
                        string(
                            credentialsId: 'minio-endpoint',
                            variable: 'MINIO_ENDPOINT'
                        )
                    ]) {
                        def zipFile = (env.TEST_TYPE == 'REGRESSION') ? env.MINIO_FILE_FULL : env.MINIO_FILE_COMPACT
                        def expectedSize = (env.TEST_TYPE == 'REGRESSION') ? '~8.5GB' : '~1.4GB'
                        
                        sh '''
                            set -e
                            mc alias set myminio "$MINIO_ENDPOINT" "$MINIO_ACCESS_KEY" "$MINIO_SECRET_KEY"
                            
                            echo "=========================================="
                            echo "Downloading ''' + zipFile + ''' (''' + expectedSize + ''')"
                            echo "=========================================="
                            mc cp myminio/${MINIO_BUCKET}/''' + zipFile + ''' .
                            
                            echo "Extracting test files..."
                            unzip -q -o ''' + zipFile + '''
                            
                            # Rename CompactTestFiles to TestFiles if needed
                            if [ -d "CompactTestFiles" ]; then
                                mv CompactTestFiles TestFiles
                            fi
                            
                            # Delete ZIP immediately to save space
                            rm -f ''' + zipFile + '''
                            
                            echo "Test files ready:"
                            find TestFiles -type f -name "*.flac" | wc -l
                            du -sh TestFiles
                        '''
                    }
                }
            }
        }
        
        stage('Build') {
            steps {
                sh '''
                    echo "Building Rust project..."
                    cargo build --release
                    
                    echo "=== Build Artifact ==="
                    ls -lh target/release/audiocheckr
                    echo "======================"
                '''
            }
        }
        
        stage('Analysis & Tests') {
            parallel {
                stage('SonarQube') {
                    when {
                        expression { return !params.SKIP_SONARQUBE }
                    }
                    stages {
                        stage('SonarQube Analysis') {
                            steps {
                                script {
                                    try {
                                        def scannerHome = tool 'SonarQube-LXC'
                                        
                                        withSonarQubeEnv('SonarQube-LXC') {
                                            sh """
                                                ${scannerHome}/bin/sonar-scanner \
                                                    -Dsonar.projectKey=${SONAR_PROJECT_KEY} \
                                                    -Dsonar.projectName=${SONAR_PROJECT_NAME} \
                                                    -Dsonar.sources=${SONAR_SOURCES} \
                                                    -Dsonar.exclusions=**/target/**,**/TestFiles/**
                                            """
                                        }
                                        echo "SonarQube analysis uploaded successfully"
                                    } catch (Exception e) {
                                        echo "SonarQube analysis failed: ${e.message}"
                                    }
                                }
                            }
                        }
                        
                        stage('Quality Gate') {
                            steps {
                                script {
                                    try {
                                        // Note: Quality Gate webhook must be configured in SonarQube
                                        // SonarQube > Project Settings > Webhooks
                                        // URL: http://JENKINS_URL/sonarqube-webhook/
                                        timeout(time: 10, unit: 'MINUTES') {
                                            def qg = waitForQualityGate abortPipeline: false
                                            if (qg.status != 'OK') {
                                                echo "Quality Gate: ${qg.status}"
                                            } else {
                                                echo "Quality Gate: PASSED"
                                            }
                                        }
                                    } catch (Exception e) {
                                        echo "Quality Gate skipped: ${e.message}"
                                        echo "Tip: Configure webhook in SonarQube > Project Settings > Webhooks"
                                        echo "URL: http://YOUR_JENKINS_URL/sonarqube-webhook/"
                                    }
                                }
                            }
                        }
                    }
                }
                
                stage('Tests') {
                    stages {
                        stage('Run Tests') {
                            steps {
                                script {
                                    def testName = (env.TEST_TYPE == 'QUALIFICATION') ? 'qualification_test' : 'regression_test'
                                    
                                    // Create test-results directory
                                    sh 'mkdir -p target/test-results'
                                    
                                    echo "=========================================="
                                    echo "Running ${env.TEST_TYPE} tests"
                                    echo "=========================================="
                                    
                                    // Run tests and capture output for JUnit XML generation
                                    def testResult = sh(
                                        script: """
                                            set +e
                                            
                                            # Run tests and capture output
                                            cargo test --test ${testName} -- --nocapture 2>&1 | tee target/test-results/test_output.txt
                                            TEST_EXIT=\${PIPESTATUS[0]}
                                            
                                            # Parse output and generate JUnit XML
                                            python3 - << 'PYTHON_SCRIPT'
import re
import sys
from xml.etree.ElementTree import Element, SubElement, tostring
from xml.dom import minidom

test_name = "${testName}"
output_file = "target/test-results/test_output.txt"
xml_file = f"target/test-results/{test_name}.xml"

try:
    with open(output_file, 'r') as f:
        content = f.read()
except:
    content = ""

# Parse test results
testsuites = Element('testsuites')
testsuite = SubElement(testsuites, 'testsuite', name=test_name)

# Find individual test results
passed = 0
failed = 0
test_cases = []

# Match patterns like "[1/19] PASS:" or "[1/19] FALSE NEGATIVE:"
pattern = r'\\[(\\d+)/(\\d+)\\]\\s+(.*?):\\s+(.*)'
matches = re.findall(pattern, content)

for match in matches:
    idx, total, status, desc = match
    testcase = SubElement(testsuite, 'testcase', 
                         name=f"test_{idx}_{desc.replace(' ', '_')[:50]}", 
                         classname=test_name)
    
    if 'PASS' in status:
        passed += 1
    else:
        failed += 1
        failure = SubElement(testcase, 'failure', message=status)
        failure.text = desc

# If no individual tests found, create summary
if not matches:
    testcase = SubElement(testsuite, 'testcase', name='test_suite', classname=test_name)
    if 'FAILED' in content or 'panicked' in content:
        failed = 1
        failure = SubElement(testcase, 'failure', message='Test suite failed')
        failure.text = content[-2000:] if len(content) > 2000 else content
    else:
        passed = 1

testsuite.set('tests', str(passed + failed))
testsuite.set('failures', str(failed))

# Write XML
xml_str = minidom.parseString(tostring(testsuites)).toprettyxml(indent="  ")
with open(xml_file, 'w') as f:
    f.write(xml_str)

print(f"Generated {xml_file}: {passed} passed, {failed} failed")
PYTHON_SCRIPT
                                            
                                            exit \$TEST_EXIT
                                        """,
                                        returnStatus: true
                                    )
                                    
                                    if (testResult != 0) {
                                        echo "Tests completed with failures (exit code: ${testResult})"
                                        echo "This is expected during development - check test results"
                                        currentBuild.result = 'UNSTABLE'
                                    } else {
                                        echo "All tests passed!"
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    
    post {
        success {
            echo 'Build and tests completed successfully!'
            archiveArtifacts artifacts: 'target/release/audiocheckr', fingerprint: true
        }
        unstable {
            echo 'Build completed but some tests failed. Check test results.'
        }
        failure {
            echo 'Build or tests failed. Check logs for details.'
        }
        always {
            // Publish JUnit test results (shows in Jenkins UI)
            junit(
                allowEmptyResults: true,
                testResults: 'target/test-results/*.xml',
                skipPublishingChecks: false
            )
            
            // Clean up everything to save disk space
            script {
                echo "Cleaning workspace to save disk space..."
                
                sh '''
                    # Delete test files and ZIPs
                    rm -f CompactTestFiles.zip TestFiles.zip
                    rm -rf CompactTestFiles TestFiles
                    
                    # Keep the release binary, clean build cache
                    if [ -f target/release/audiocheckr ]; then
                        cp target/release/audiocheckr /tmp/audiocheckr_backup_$BUILD_NUMBER 2>/dev/null || true
                    fi
                    
                    # Clean target directory (saves ~2GB+)
                    rm -rf target/debug
                    rm -rf target/release/deps
                    rm -rf target/release/build
                    rm -rf target/release/.fingerprint
                    rm -rf target/release/incremental
                    
                    # Restore binary
                    if [ -f /tmp/audiocheckr_backup_$BUILD_NUMBER ]; then
                        mkdir -p target/release
                        mv /tmp/audiocheckr_backup_$BUILD_NUMBER target/release/audiocheckr
                    fi
                    
                    echo "Cleanup complete"
                    du -sh . 2>/dev/null || true
                '''
            }
        }
    }
}