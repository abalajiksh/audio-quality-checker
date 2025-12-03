pipeline {
    agent any

    environment {
        RUST_BACKTRACE = '1'
        CARGO_HOME = "${WORKSPACE}/.cargo"
        PATH = "${WORKSPACE}/.cargo/bin:${env.PATH}:/var/lib/jenkins/bin:/var/lib/jenkins/.cargo/bin:$HOME/bin"
        RUSTUP_HOME = '/var/lib/jenkins/.rustup'
        
        // MinIO configuration
        MINIO_BUCKET = 'audiocheckr'
        MINIO_FILE_COMPACT = 'CompactTestFiles.zip'
        MINIO_FILE_FULL = 'TestFiles.zip'
        MINIO_FILE_GENRE_LITE = 'GenreTestSuiteLite.zip'
        MINIO_FILE_GENRE_FULL = 'TestSuite.zip'
    }

    parameters {
        choice(
            name: 'TEST_TYPE_OVERRIDE',
            choices: ['AUTO', 'QUALIFICATION', 'REGRESSION', 'DIAGNOSTIC'],
            description: 'Override automatic test type selection'
        )
        booleanParam(
            name: 'BUILD_ARM64',
            defaultValue: true,
            description: 'Build ARM64 binary (cross-compilation)'
        )
        booleanParam(
            name: 'CLEAN_WORKSPACE_BEFORE',
            defaultValue: false,
            description: 'Clean workspace before build'
        )
    }

    triggers {
        // Run regression tests weekly on Sunday at 2 AM
        cron('0 2 * * 0')
    }

    options {
        buildDiscarder(logRotator(
            numToKeepStr: '10',
            artifactNumToKeepStr: '5'))
        timestamps()
        disableConcurrentBuilds()
        timeout(time: 90, unit: 'MINUTES')
    }

    stages {
        stage('Pre-flight') {
            steps {
                script {
                    // Determine test type
                    if (params.TEST_TYPE_OVERRIDE && params.TEST_TYPE_OVERRIDE != 'AUTO') {
                        env.TEST_TYPE = params.TEST_TYPE_OVERRIDE
                        echo "🔧 Test type forced via parameter: ${env.TEST_TYPE}"
                    } else if (currentBuild.getBuildCauses('hudson.triggers.TimerTrigger$TimerTriggerCause')) {
                        env.TEST_TYPE = 'REGRESSION'
                        echo "⏰ Scheduled build detected - running REGRESSION tests"
                    } else if (currentBuild.getBuildCauses('hudson.model.Cause$UserIdCause')) {
                        env.TEST_TYPE = 'QUALIFICATION'
                        echo "👤 Manual build - running QUALIFICATION tests (use parameter to override)"
                    } else {
                        env.TEST_TYPE = 'QUALIFICATION'
                        echo "🔄 Push detected - running QUALIFICATION tests"
                    }

                    echo """
========================================================
                  AUDIOCHECKR CI/CD                     
========================================================
  Test Type:     ${env.TEST_TYPE}
  ARM Build:     ${params.BUILD_ARM64 ? 'ENABLED ✓' : 'DISABLED'}
  Build #:       ${currentBuild.number}
  Triggered by:  ${currentBuild.getBuildCauses()[0].shortDescription}
========================================================
"""
                }
            }
        }

        stage('Setup & Checkout') {
            parallel {
                stage('Setup Tools') {
                    steps {
                        sh '''
                            mkdir -p $HOME/bin
                            
                            if ! command -v cc >/dev/null 2>&1; then
                                echo "ERROR: C compiler not found!"
                                exit 1
                            fi
                            
                            # Check for MinIO client specifically (not Midnight Commander!)
                            # We use a dedicated path to avoid conflicts
                            if [ ! -f "$HOME/bin/minio-mc" ]; then
                                echo "Installing MinIO client..."
                                wget -q https://dl.min.io/client/mc/release/linux-amd64/mc -O $HOME/bin/minio-mc
                                chmod +x $HOME/bin/minio-mc
                            fi
                            
                            # Verify it's actually the MinIO client
                            if ! $HOME/bin/minio-mc --version 2>&1 | grep -q "RELEASE"; then
                                echo "ERROR: MinIO client not working correctly, re-downloading..."
                                rm -f $HOME/bin/minio-mc
                                wget -q https://dl.min.io/client/mc/release/linux-amd64/mc -O $HOME/bin/minio-mc
                                chmod +x $HOME/bin/minio-mc
                            fi
                            
                            if ! command -v cargo >/dev/null 2>&1; then
                                echo "ERROR: Cargo not found!"
                                exit 1
                            fi
                            
                            echo "=== Tool Versions ==="
                            echo "MinIO Client:"
                            $HOME/bin/minio-mc --version
                            echo ""
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
            }
        }

        stage('Build & Prepare') {
            parallel {
                stage('Build x86_64') {
                    steps {
                        sh '''
                            echo "=========================================="
                            echo "Building for x86_64 (native)"
                            echo "=========================================="
                            cargo build --release 2>&1 | tee build_x86_64.txt
                            
                            if grep -q "warning:" build_x86_64.txt; then
                                echo ""
                                echo "⚠️  Build completed with warnings"
                            else
                                echo ""
                                echo "✓ Build completed without warnings"
                            fi
                            
                            echo ""
                            echo "=== Build Artifact ==="
                            ls -lh target/release/audiocheckr
                            file target/release/audiocheckr
                            echo "======================"
                        '''
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
                                if (env.TEST_TYPE == 'DIAGNOSTIC') {
                                    sh '''
                                        set -e
                                        MC="$HOME/bin/minio-mc"
                                        
                                        echo "Setting up MinIO alias..."
                                        $MC alias set myminio "$MINIO_ENDPOINT" "$MINIO_ACCESS_KEY" "$MINIO_SECRET_KEY"
                                        
                                        echo "=========================================="
                                        echo "Downloading DIAGNOSTIC test files"
                                        echo "=========================================="
                                        
                                        # Download and extract TestSuite only
                                        echo "Downloading ${MINIO_FILE_GENRE_FULL}"
                                        $MC cp myminio/${MINIO_BUCKET}/${MINIO_FILE_GENRE_FULL} .
                                        unzip -q -o ${MINIO_FILE_GENRE_FULL}
                                        rm -f ${MINIO_FILE_GENRE_FULL}
                                        
                                        echo "✓ Test files ready for diagnostic"
                                        ls -la
                                    '''
                                } else if (env.TEST_TYPE == 'REGRESSION') {
                                    sh '''
                                        set -e
                                        MC="$HOME/bin/minio-mc"
                                        
                                        echo "Setting up MinIO alias..."
                                        $MC alias set myminio "$MINIO_ENDPOINT" "$MINIO_ACCESS_KEY" "$MINIO_SECRET_KEY"
                                        
                                        echo "=========================================="
                                        echo "Downloading REGRESSION test files"
                                        echo "=========================================="
                                        
                                        # Download and extract TestFiles
                                        echo "Downloading ${MINIO_FILE_FULL}"
                                        $MC cp myminio/${MINIO_BUCKET}/${MINIO_FILE_FULL} .
                                        unzip -q -o ${MINIO_FILE_FULL}
                                        rm -f ${MINIO_FILE_FULL}
                                        
                                        # Download and extract TestSuite
                                        echo "Downloading ${MINIO_FILE_GENRE_FULL}"
                                        $MC cp myminio/${MINIO_BUCKET}/${MINIO_FILE_GENRE_FULL} .
                                        unzip -q -o ${MINIO_FILE_GENRE_FULL}
                                        rm -f ${MINIO_FILE_GENRE_FULL}
                                        
                                        echo "✓ Test files ready"
                                        ls -la
                                    '''
                                } else {
                                    sh '''
                                        set -e
                                        MC="$HOME/bin/minio-mc"
                                        
                                        echo "Setting up MinIO alias..."
                                        $MC alias set myminio "$MINIO_ENDPOINT" "$MINIO_ACCESS_KEY" "$MINIO_SECRET_KEY"
                                        
                                        echo "=========================================="
                                        echo "Downloading QUALIFICATION test files"
                                        echo "=========================================="
                                        
                                        # Download and extract CompactTestFiles
                                        echo "Downloading ${MINIO_FILE_COMPACT}"
                                        $MC cp myminio/${MINIO_BUCKET}/${MINIO_FILE_COMPACT} .
                                        unzip -q -o ${MINIO_FILE_COMPACT}
                                        if [ -d "CompactTestFiles" ]; then
                                            mv CompactTestFiles TestFiles
                                        fi
                                        rm -f ${MINIO_FILE_COMPACT}
                                        
                                        # Download and extract GenreTestSuiteLite
                                        echo "Downloading ${MINIO_FILE_GENRE_LITE}"
                                        $MC cp myminio/${MINIO_BUCKET}/${MINIO_FILE_GENRE_LITE} .
                                        unzip -q -o ${MINIO_FILE_GENRE_LITE}
                                        rm -f ${MINIO_FILE_GENRE_LITE}
                                        
                                        echo "✓ Test files ready"
                                        ls -la
                                    '''
                                }
                            }
                        }
                    }
                }
            }
        }

        stage('Build ARM64') {
            when {
                expression { params.BUILD_ARM64 }
            }
            steps {
                sh '''
                    echo "=========================================="
                    echo "Building for ARM64 (cross-compilation)"
                    echo "=========================================="
                    
                    # Add ARM64 target if not present
                    rustup target add aarch64-unknown-linux-gnu 2>/dev/null || true
                    
                    # Build with cross-compiler
                    CARGO_TARGET_AARCH64_UNKNOWN_LINUX_GNU_LINKER=aarch64-linux-gnu-gcc \
                    cargo build --release --target aarch64-unknown-linux-gnu 2>&1 | tee build_arm64.txt
                    
                    if ! grep -q "warning:" build_arm64.txt; then
                        echo ""
                        echo "✓ ARM64 build completed without warnings"
                    fi
                    
                    mkdir -p target/arm64
                    cp target/aarch64-unknown-linux-gnu/release/audiocheckr target/arm64/audiocheckr-arm64
                    
                    echo ""
                    echo "=== ARM64 Build Artifact ==="
                    ls -lh target/arm64/audiocheckr-arm64
                    file target/arm64/audiocheckr-arm64
                    echo "============================="
                '''
            }
        }

        stage('ARM64 Binary Validation') {
            when {
                expression { params.BUILD_ARM64 }
            }
            steps {
                script {
                    sh '''
                        echo "Validating ARM64 binary..."
                        file target/arm64/audiocheckr-arm64 | grep -q "aarch64"
                        echo "✓ ARM64 binary validation passed"
                    '''
                }
            }
        }

        stage('Run Tests') {
            steps {
                script {
                    switch(env.TEST_TYPE) {
                        case 'QUALIFICATION':
                            // Run both qualification tests together
                            sh '''
                                echo ""
                                echo "=========================================="
                                echo "Running QUALIFICATION Tests"
                                echo "=========================================="
                                echo ""
                                
                                # Run qualification_test (uses TestFiles)
                                echo ">>> Running qualification_test..."
                                cargo test --release --test qualification_test -- --nocapture --test-threads=1 2>&1 | tee qualification_results.txt
                                QUALI_EXIT=${PIPESTATUS[0]}
                                
                                echo ""
                                echo "=========================================="
                                echo ""
                                
                                # Run qualification_genre_test (uses GenreTestSuiteLite or TestSuite)
                                echo ">>> Running qualification_genre_test..."
                                cargo test --release --test qualification_genre_test -- --nocapture --test-threads=1 2>&1 | tee qualification_genre_results.txt
                                QUALI_GENRE_EXIT=${PIPESTATUS[0]}
                                
                                echo ""
                                echo "=========================================="
                                echo "QUALIFICATION TEST SUMMARY"
                                echo "=========================================="
                                
                                if [ $QUALI_EXIT -eq 0 ]; then
                                    echo "✓ qualification_test: PASSED"
                                else
                                    echo "✗ qualification_test: FAILED"
                                fi
                                
                                if [ $QUALI_GENRE_EXIT -eq 0 ]; then
                                    echo "✓ qualification_genre_test: PASSED"
                                else
                                    echo "✗ qualification_genre_test: FAILED"
                                fi
                                
                                echo "=========================================="
                                
                                # Fail if either test failed
                                if [ $QUALI_EXIT -ne 0 ] || [ $QUALI_GENRE_EXIT -ne 0 ]; then
                                    exit 1
                                fi
                            '''
                            break

                        case 'REGRESSION':
                            sh '''
                                echo ""
                                echo "=========================================="
                                echo "Running REGRESSION Tests (Full Suite)"
                                echo "=========================================="
                                echo ""
                                
                                # Run regression_test (uses TestFiles)
                                echo ">>> Running regression_test..."
                                cargo test --release --test regression_test -- --nocapture --test-threads=1 2>&1 | tee regression_results.txt
                                REG_EXIT=${PIPESTATUS[0]}
                                
                                echo ""
                                echo "=========================================="
                                echo ""
                                
                                # Run regression_genre_test (uses full TestSuite)
                                echo ">>> Running regression_genre_test..."
                                cargo test --release --test regression_genre_test -- --nocapture --test-threads=1 2>&1 | tee regression_genre_results.txt
                                REG_GENRE_EXIT=${PIPESTATUS[0]}
                                
                                echo ""
                                echo "=========================================="
                                echo "REGRESSION TEST SUMMARY"
                                echo "=========================================="
                                
                                if [ $REG_EXIT -eq 0 ]; then
                                    echo "✓ regression_test: PASSED"
                                else
                                    echo "✗ regression_test: FAILED"
                                fi
                                
                                if [ $REG_GENRE_EXIT -eq 0 ]; then
                                    echo "✓ regression_genre_test: PASSED"
                                else
                                    echo "✗ regression_genre_test: FAILED"
                                fi
                                
                                echo "=========================================="
                                
                                # For regression tests, we report but don't fail the build
                                # This allows tracking improvements over time
                                # Uncomment next lines to enforce strict pass/fail:
                                # if [ $REG_EXIT -ne 0 ] || [ $REG_GENRE_EXIT -ne 0 ]; then
                                #     exit 1
                                # fi
                            '''
                            break

                        case 'DIAGNOSTIC':
                            sh '''
                                echo ""
                                echo "=========================================="
                                echo "Running DIAGNOSTIC Tests"
                                echo "=========================================="
                                echo ""
                                
                                # Run the full regression genre test for diagnostic
                                echo ">>> Running regression_genre_test (diagnostic mode)..."
                                cargo test --release --test regression_genre_test -- --nocapture --test-threads=1 2>&1 | tee diagnostic_results.txt
                                
                                echo ""
                                echo "=========================================="
                                echo "DIAGNOSTIC COMPLETE"
                                echo "=========================================="
                            '''
                            break

                        default:
                            error "Unknown test type: ${env.TEST_TYPE}"
                    }
                }
            }
        }
    }

    post {
        always {
            script {
                // Archive build artifacts
                archiveArtifacts artifacts: 'target/release/audiocheckr', fingerprint: true, allowEmptyArchive: true
                archiveArtifacts artifacts: 'target/arm64/audiocheckr-arm64', fingerprint: true, allowEmptyArchive: true
                archiveArtifacts artifacts: '*_results.txt', fingerprint: true, allowEmptyArchive: true

                // Try to publish test results
                junit allowEmptyResults: true, testResults: 'target/nextest/ci/junit.xml'

                // Cleanup to save disk space
                echo "🧹 Cleaning workspace to save disk space..."
                sh '''
                    # Remove downloaded test files
                    rm -f CompactTestFiles.zip TestFiles.zip GenreTestSuiteLite.zip TestSuite.zip
                    rm -rf CompactTestFiles TestFiles GenreTestSuiteLite TestSuite
                    
                    # Keep binaries but clean build artifacts
                    if [ -f target/release/audiocheckr ]; then
                        cp target/release/audiocheckr /tmp/audiocheckr_backup_x86_${BUILD_NUMBER}
                    fi
                    if [ -f target/arm64/audiocheckr-arm64 ]; then
                        cp target/arm64/audiocheckr-arm64 /tmp/audiocheckr_backup_arm64_${BUILD_NUMBER}
                    fi
                    
                    rm -rf target/debug
                    rm -rf target/release/deps
                    rm -rf target/release/build
                    rm -rf target/release/.fingerprint
                    rm -rf target/release/incremental
                    rm -rf target/aarch64-unknown-linux-gnu/release/deps
                    rm -rf target/aarch64-unknown-linux-gnu/release/build
                    rm -rf target/aarch64-unknown-linux-gnu/release/.fingerprint
                    rm -rf target/aarch64-unknown-linux-gnu/release/incremental
                    
                    # Restore binaries
                    if [ -f /tmp/audiocheckr_backup_x86_${BUILD_NUMBER} ]; then
                        mkdir -p target/release
                        mv /tmp/audiocheckr_backup_x86_${BUILD_NUMBER} target/release/audiocheckr
                    fi
                    if [ -f /tmp/audiocheckr_backup_arm64_${BUILD_NUMBER} ]; then
                        mkdir -p target/arm64
                        mv /tmp/audiocheckr_backup_arm64_${BUILD_NUMBER} target/arm64/audiocheckr-arm64
                    fi
                    
                    echo "✓ Cleanup complete"
                    du -sh .
                '''
            }
        }
        success {
            echo "✅ Build and tests completed successfully!"
        }
        failure {
            echo "❌ Build or tests failed!"
        }
    }
}
