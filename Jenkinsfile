pipeline {
    agent any

    options {
        timestamps()
        timeout(time: 90, unit: 'MINUTES')
        buildDiscarder(logRotator(numToKeepStr: '10'))
    }

    parameters {
        choice(
            name: 'TEST_TYPE',
            choices: ['AUTO', 'QUALIFICATION', 'REGRESSION', 'DIAGNOSTIC'],
            description: 'Test type: AUTO detects from trigger, others force specific test'
        )
        booleanParam(
            name: 'ENABLE_ARM64',
            defaultValue: true,
            description: 'Enable ARM64 cross-compilation builds'
        )
        booleanParam(
            name: 'SKIP_SONARQUBE',
            defaultValue: false,
            description: 'Skip SonarQube analysis'
        )
    }

    environment {
        RUST_BACKTRACE = '1'
        CARGO_TERM_COLOR = 'never'
        PATH = "$HOME/bin:$HOME/.cargo/bin:/usr/bin:$PATH"
    }

    stages {
        stage('Pre-flight') {
            steps {
                script {
                    // Determine test type
                    if (params.TEST_TYPE == 'AUTO') {
                        if (currentBuild.getBuildCauses('hudson.model.Cause$UserIdCause').size() > 0) {
                            env.EFFECTIVE_TEST_TYPE = 'QUALIFICATION'
                            echo "👤 Manual trigger detected - running QUALIFICATION tests"
                        } else if (currentBuild.getBuildCauses('com.cloudbees.jenkins.GitHubPushCause').size() > 0 ||
                                   currentBuild.getBuildCauses('hudson.triggers.SCMTrigger$SCMTriggerCause').size() > 0) {
                            env.EFFECTIVE_TEST_TYPE = 'QUALIFICATION'
                            echo "🔄 Push detected - running QUALIFICATION tests"
                        } else if (currentBuild.getBuildCauses('hudson.triggers.TimerTrigger$TimerTriggerCause').size() > 0) {
                            env.EFFECTIVE_TEST_TYPE = 'REGRESSION'
                            echo "⏰ Scheduled trigger - running REGRESSION tests"
                        } else {
                            env.EFFECTIVE_TEST_TYPE = 'QUALIFICATION'
                            echo "❓ Unknown trigger - defaulting to QUALIFICATION tests"
                        }
                    } else {
                        env.EFFECTIVE_TEST_TYPE = params.TEST_TYPE
                        echo "🔧 Test type forced via parameter: ${env.EFFECTIVE_TEST_TYPE}"
                    }

                    echo """
========================================================
                  AUDIOCHECKR CI/CD                     
========================================================
  Test Type:     ${env.EFFECTIVE_TEST_TYPE}
  ARM Build:     ${params.ENABLE_ARM64 ? 'ENABLED ✓' : 'DISABLED ⏭️'}
  Build #:       ${env.BUILD_NUMBER}
  Triggered by:  ${currentBuild.getBuildCauses()[0]?.shortDescription ?: 'Unknown'}
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
                            
                            # Check for warnings (informational only)
                            if grep -q "warning:" build_x86_64.txt; then
                                echo ""
                                echo "⚠️  Build completed with warnings (see above)"
                            else
                                echo ""
                                echo "✓ Build completed without warnings"
                            fi
                            
                            echo ""
                            echo "=== x86_64 Build Artifact ==="
                            ls -lh target/release/audiocheckr
                            file target/release/audiocheckr
                            echo "============================="
                        '''
                    }
                }
                stage('Download Test Files') {
                    steps {
                        script {
                            withCredentials([
                                string(credentialsId: 'minio-endpoint', variable: 'MINIO_ENDPOINT'),
                                string(credentialsId: 'minio-secret-key', variable: 'MINIO_SECRET_KEY')
                            ]) {
                                sh """
                                    set -e
                                    mc alias set myminio \${MINIO_ENDPOINT} uyBezAIGJMDw7MZH1xEt \${MINIO_SECRET_KEY}
                                    
                                    echo "=========================================="
                                    echo "Downloading ${env.EFFECTIVE_TEST_TYPE} test files"
                                    echo "=========================================="
                                    
                                    case "${env.EFFECTIVE_TEST_TYPE}" in
                                        QUALIFICATION)
                                            echo "Downloading CompactTestFiles.zip (~1.4GB)"
                                            mc cp myminio/audiocheckr/CompactTestFiles.zip .
                                            unzip -q -o CompactTestFiles.zip
                                            rm -f CompactTestFiles.zip
                                            echo "Test files ready"
                                            find CompactTestFiles -type f -name "*.flac" | wc -l
                                            du -sh CompactTestFiles
                                            ;;
                                        REGRESSION)
                                            echo "Downloading TestFiles.zip (~8.5GB)"
                                            mc cp myminio/audiocheckr/TestFiles.zip .
                                            unzip -q -o TestFiles.zip
                                            rm -f TestFiles.zip
                                            echo "Test files ready"
                                            find TestFiles -type f -name "*.flac" | wc -l
                                            du -sh TestFiles
                                            ;;
                                        DIAGNOSTIC)
                                            echo "Downloading TestSuite.zip (~19.4GB)"
                                            mc cp myminio/audiocheckr/TestSuite.zip .
                                            unzip -q -o TestSuite.zip
                                            rm -f TestSuite.zip
                                            echo "Test files ready for diagnostic"
                                            find TestSuite -type f -name "*.flac" | wc -l
                                            du -sh TestSuite
                                            ;;
                                    esac
                                """
                            }
                        }
                    }
                }
            }
        }

        stage('SonarQube Analysis') {
            when {
                allOf {
                    expression { return !params.SKIP_SONARQUBE }
                    expression { return env.EFFECTIVE_TEST_TYPE != 'DIAGNOSTIC' }
                }
            }
            steps {
                script {
                    withSonarQubeEnv('SonarQube-LXC') {
                        sh '''
                            sonar-scanner \
                                -Dsonar.projectKey=audiocheckr \
                                -Dsonar.projectName="AudioCheckr" \
                                -Dsonar.sources=src \
                                -Dsonar.tests=tests \
                                -Dsonar.language=rust \
                                -Dsonar.exclusions="**/target/**,**/TestFiles/**,**/TestSuite/**,**/GenreTestSuiteLite/**"
                        '''
                    }
                    echo "✓ SonarQube analysis completed"
                }
            }
        }

        stage('Quality Gate') {
            when {
                allOf {
                    expression { return !params.SKIP_SONARQUBE }
                    expression { return env.EFFECTIVE_TEST_TYPE != 'DIAGNOSTIC' }
                }
            }
            steps {
                script {
                    timeout(time: 10, unit: 'MINUTES') {
                        def qg = waitForQualityGate()
                        echo "⚠ Quality Gate: ${qg.status}"
                    }
                }
            }
        }

        stage('Diagnostic Test') {
            when {
                expression { return env.EFFECTIVE_TEST_TYPE == 'DIAGNOSTIC' }
            }
            steps {
                script {
                    echo "=========================================="
                    echo "Running DIAGNOSTIC TEST"
                    echo "=========================================="
                    
                    sh 'mkdir -p target/test-results'
                    
                    def testResult = sh(
                        script: '''
                            set +e
                            cargo test --test diagnostic_test -- --nocapture 2>&1 | tee target/test-results/diagnostic.txt
                            exit $?
                        ''',
                        returnStatus: true
                    )
                    
                    if (testResult == 0) {
                        echo "✓ Diagnostic test passed!"
                    } else {
                        echo "⚠ Diagnostic test completed with findings"
                    }
                    
                    archiveArtifacts artifacts: 'target/test-results/diagnostic.txt', allowEmptyArchive: true
                }
            }
        }

        stage('x86_64 Tests (Full Suite)') {
            when {
                expression { return env.EFFECTIVE_TEST_TYPE != 'DIAGNOSTIC' }
            }
            stages {
                stage('Integration Tests') {
                    steps {
                        script {
                            echo "=========================================="
                            echo "x86_64: Integration Tests"
                            echo "=========================================="
                            
                            sh 'mkdir -p target/test-results'
                            
                            def testResult = sh(
                                script: '''
                                    set +e
                                    cargo test --test integration_test -- --nocapture 2>&1 | tee target/test-results/integration_x86_64.txt
                                    exit $?
                                ''',
                                returnStatus: true
                            )
                            
                            if (testResult == 0) {
                                echo "✓ x86_64 integration tests passed!"
                            } else {
                                echo "⚠ x86_64 integration tests had failures"
                                currentBuild.result = 'UNSTABLE'
                            }
                        }
                    }
                }

                stage('Qualification Tests') {
                    when {
                        expression { return env.EFFECTIVE_TEST_TYPE == 'QUALIFICATION' }
                    }
                    parallel {
                        stage('Qualification Test') {
                            steps {
                                script {
                                    echo "=========================================="
                                    echo "x86_64: Qualification Tests"
                                    echo "=========================================="
                                    
                                    def testResult = sh(
                                        script: '''
                                            set +e
                                            cargo test --test qualification_test -- --nocapture 2>&1 | tee target/test-results/qualification_x86_64.txt
                                            exit $?
                                        ''',
                                        returnStatus: true
                                    )
                                    
                                    if (testResult == 0) {
                                        echo "✓ x86_64 qualification tests passed!"
                                    } else {
                                        echo "⚠ x86_64 qualification tests had failures"
                                        currentBuild.result = 'UNSTABLE'
                                    }
                                }
                            }
                        }
                        stage('Qualification Genre Test') {
                            steps {
                                script {
                                    def testResult = sh(
                                        script: '''
                                            set +e
                                            cargo test --test qualification_genre_test -- --nocapture 2>&1 | tee target/test-results/qualification_genre_x86_64.txt
                                            exit $?
                                        ''',
                                        returnStatus: true
                                    )
                                    
                                    if (testResult == 0) {
                                        echo "✓ x86_64 qualification genre tests passed!"
                                    } else {
                                        echo "⚠ x86_64 qualification genre tests had failures"
                                        currentBuild.result = 'UNSTABLE'
                                    }
                                }
                            }
                        }
                    }
                }

                stage('Regression Tests') {
                    when {
                        expression { return env.EFFECTIVE_TEST_TYPE == 'REGRESSION' }
                    }
                    steps {
                        script {
                            echo "=========================================="
                            echo "x86_64: Regression Tests"
                            echo "=========================================="
                            
                            def testResult = sh(
                                script: '''
                                    set +e
                                    cargo test --release --test regression_test -- --nocapture 2>&1 | tee target/test-results/regression_x86_64.txt
                                    exit $?
                                ''',
                                returnStatus: true
                            )
                            
                            if (testResult == 0) {
                                echo "✓ x86_64 regression tests passed!"
                            } else {
                                echo "⚠ Regression tests completed with failures"
                                currentBuild.result = 'UNSTABLE'
                            }
                        }
                    }
                }

                stage('Regression Genre Tests') {
                    when {
                        expression { return env.EFFECTIVE_TEST_TYPE == 'REGRESSION' }
                    }
                    steps {
                        script {
                            def testResult = sh(
                                script: '''
                                    set +e
                                    cargo test --release --test regression_genre_test -- --nocapture 2>&1 | tee target/test-results/regression_genre_x86_64.txt
                                    exit $?
                                ''',
                                returnStatus: true
                            )
                            
                            if (testResult == 0) {
                                echo "✓ x86_64 regression genre tests passed!"
                            } else {
                                echo "⚠ Regression genre tests completed with failures"
                                currentBuild.result = 'UNSTABLE'
                            }
                        }
                    }
                }
            }
        }

        stage('ARM64 Validation (Cross-Compile)') {
            when {
                allOf {
                    expression { return params.ENABLE_ARM64 }
                    expression { return env.EFFECTIVE_TEST_TYPE != 'DIAGNOSTIC' }
                }
            }
            stages {
                stage('ARM64 Build') {
                    steps {
                        sh '''
                            echo "=========================================="
                            echo "Building for ARM64 (cross-compile)"
                            echo "=========================================="
                            
                            # Ensure we have the ARM64 target
                            rustup target add aarch64-unknown-linux-gnu 2>/dev/null || true
                            
                            # Build for ARM64 - release only (no tests compilation on different arch)
                            CARGO_TARGET_AARCH64_UNKNOWN_LINUX_GNU_LINKER=aarch64-linux-gnu-gcc \
                            cargo build --release --target aarch64-unknown-linux-gnu 2>&1 | tee build_arm64.txt
                            
                            # Check for warnings
                            if grep -q "warning:" build_arm64.txt; then
                                echo ""
                                echo "⚠️  ARM64 build completed with warnings"
                            else
                                echo ""
                                echo "✓ ARM64 build completed without warnings"
                            fi
                            
                            # Copy ARM64 binary to accessible location
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
                    steps {
                        script {
                            sh '''
                                echo "Validating ARM64 binary..."
                                
                                # Verify it's actually an ARM64 binary
                                file target/arm64/audiocheckr-arm64 | grep -q "aarch64" || {
                                    echo "ERROR: Binary is not ARM64!"
                                    exit 1
                                }
                                
                                echo "✓ ARM64 binary validation passed"
                            '''
                        }
                    }
                }
            }
        }
    }

    post {
        always {
            script {
                // Archive build artifacts
                archiveArtifacts artifacts: 'target/release/audiocheckr', allowEmptyArchive: true, fingerprint: true
                
                if (params.ENABLE_ARM64 && env.EFFECTIVE_TEST_TYPE != 'DIAGNOSTIC') {
                    archiveArtifacts artifacts: 'target/arm64/audiocheckr-arm64', allowEmptyArchive: true, fingerprint: true
                }
                
                // Archive build logs
                archiveArtifacts artifacts: 'build_*.txt', allowEmptyArchive: true
            }

            // Collect test results
            junit testResults: 'target/test-results/*.xml', allowEmptyResults: true

            script {
                echo "🧹 Cleaning workspace to save disk space..."
                sh '''
                    # Remove downloaded test files
                    rm -f CompactTestFiles.zip TestFiles.zip GenreTestSuiteLite.zip TestSuite.zip
                    rm -rf CompactTestFiles TestFiles GenreTestSuiteLite TestSuite
                    
                    # Backup important binaries
                    if [ -f target/release/audiocheckr ]; then
                        cp target/release/audiocheckr /tmp/audiocheckr_backup_x86_${BUILD_NUMBER}
                    fi
                    if [ -f target/arm64/audiocheckr-arm64 ]; then
                        cp target/arm64/audiocheckr-arm64 /tmp/audiocheckr_backup_arm64_${BUILD_NUMBER}
                    fi
                    
                    # Clean build directories but keep binaries
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

        unstable {
            echo "⚠ Build completed but some tests failed. Check test results."
        }

        failure {
            echo "❌ Build failed! Check the logs for details."
        }
    }
}
