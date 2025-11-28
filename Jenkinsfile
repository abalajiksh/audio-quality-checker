pipeline {
    agent any
    
    // Build parameters - allows manual trigger with options
    parameters {
        choice(
            name: 'TEST_TYPE_OVERRIDE',
            choices: ['AUTO', 'QUALIFICATION', 'REGRESSION', 'REGRESSION_GENRE'],
            description: 'Force a specific test type. AUTO uses smart detection.'
        )
        booleanParam(
            name: 'RUN_GENRE_REGRESSION',
            defaultValue: false,
            description: 'Run regression genre tests (manual trigger only)'
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
        MINIO_FILE_GENRE_LITE = 'GenreTestSuiteLite.zip'
        MINIO_FILE_GENRE_FULL = 'TestSuite.zip'
        
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
                stage('Build') {
                    steps {
                        sh '''
                            echo "Building Rust project..."
                            cargo build --release 2>&1 | tee build_output.txt
                            
                            # Check for warnings (informational, doesn't fail build)
                            if grep -q "warning:" build_output.txt; then
                                echo ""
                                echo "=== Build Warnings Summary ==="
                                grep -c "warning:" build_output.txt || true
                                echo "warnings found (see above for details)"
                                echo "=============================="
                            fi
                            
                            echo ""
                            echo "=== Build Artifact ==="
                            ls -lh target/release/audiocheckr
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
                                if (env.TEST_TYPE == 'REGRESSION' || env.TEST_TYPE == 'REGRESSION_GENRE') {
                                    sh '''
                                        set -e
                                        mc alias set myminio "$MINIO_ENDPOINT" "$MINIO_ACCESS_KEY" "$MINIO_SECRET_KEY"
                                        
                                        echo "=========================================="
                                        echo "Downloading REGRESSION test files"
                                        echo "=========================================="
                                        
                                        # Download and extract TestFiles
                                        echo "Downloading ''' + env.MINIO_FILE_FULL + ''' (~8.5GB)"
                                        mc cp myminio/${MINIO_BUCKET}/''' + env.MINIO_FILE_FULL + ''' .
                                        unzip -q -o ''' + env.MINIO_FILE_FULL + '''
                                        rm -f ''' + env.MINIO_FILE_FULL + '''
                                        
                                        # Download and extract TestSuite
                                        echo "Downloading ''' + env.MINIO_FILE_GENRE_FULL + ''' (~2.5GB)"
                                        mc cp myminio/${MINIO_BUCKET}/''' + env.MINIO_FILE_GENRE_FULL + ''' .
                                        unzip -q -o ''' + env.MINIO_FILE_GENRE_FULL + '''
                                        rm -f ''' + env.MINIO_FILE_GENRE_FULL + '''
                                        
                                        echo "Test files ready"
                                        find TestFiles TestSuite -type f -name "*.flac" 2>/dev/null | wc -l || echo "0"
                                        du -sh TestFiles TestSuite 2>/dev/null || true
                                    '''
                                } else {
                                    sh '''
                                        set -e
                                        mc alias set myminio "$MINIO_ENDPOINT" "$MINIO_ACCESS_KEY" "$MINIO_SECRET_KEY"
                                        
                                        echo "=========================================="
                                        echo "Downloading QUALIFICATION test files"
                                        echo "=========================================="
                                        
                                        # Download and extract CompactTestFiles
                                        echo "Downloading ''' + env.MINIO_FILE_COMPACT + ''' (~1.4GB)"
                                        mc cp myminio/${MINIO_BUCKET}/''' + env.MINIO_FILE_COMPACT + ''' .
                                        unzip -q -o ''' + env.MINIO_FILE_COMPACT + '''
                                        if [ -d "CompactTestFiles" ]; then
                                            mv CompactTestFiles TestFiles
                                        fi
                                        rm -f ''' + env.MINIO_FILE_COMPACT + '''
                                        
                                        # Download and extract GenreTestSuiteLite
                                        echo "Downloading ''' + env.MINIO_FILE_GENRE_LITE + ''' (~800MB)"
                                        mc cp myminio/${MINIO_BUCKET}/''' + env.MINIO_FILE_GENRE_LITE + ''' .
                                        unzip -q -o ''' + env.MINIO_FILE_GENRE_LITE + '''
                                        rm -f ''' + env.MINIO_FILE_GENRE_LITE + '''
                                        
                                        echo "Test files ready"
                                        find TestFiles GenreTestSuiteLite -type f -name "*.flac" 2>/dev/null | wc -l || echo "0"
                                        du -sh TestFiles GenreTestSuiteLite 2>/dev/null || true
                                    '''
                                }
                            }
                        }
                    }
                }
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
                                                    -Dsonar.exclusions=**/target/**,**/TestFiles/**,**/TestSuite/**,**/GenreTestSuiteLite/**
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
                        stage('Integration Tests') {
                            steps {
                                script {
                                    echo "=========================================="
                                    echo "Running Integration Tests"
                                    echo "=========================================="
                                    
                                    sh 'mkdir -p target/test-results'
                                    
                                    def integrationResult = sh(
                                        script: '''
                                            set +e
                                            cargo test --test integration_test -- --nocapture 2>&1 | tee target/test-results/integration_output.txt
                                            exit ${PIPESTATUS[0]}
                                        ''',
                                        returnStatus: true
                                    )
                                    
                                    if (integrationResult != 0) {
                                        echo "Integration tests had failures (exit code: ${integrationResult})"
                                    } else {
                                        echo "Integration tests passed!"
                                    }
                                }
                            }
                        }
                        
                        stage('Run Test Suites') {
                            parallel {
                                stage('Qualification Tests') {
                                    when {
                                        expression { return env.TEST_TYPE == 'QUALIFICATION' }
                                    }
                                    steps {
                                        script {
                                            sh 'mkdir -p target/test-results'
                                            
                                            echo "=========================================="
                                            echo "Running QUALIFICATION tests"
                                            echo "=========================================="
                                            
                                            def testResult = sh(
                                                script: """
                                                    set +e
                                                    cargo test --test qualification_test -- --nocapture 2>&1 | tee target/test-results/qualification_output.txt
                                                    exit \${PIPESTATUS[0]}
                                                """,
                                                returnStatus: true
                                            )
                                            
                                            if (testResult != 0) {
                                                echo "Qualification tests completed with failures"
                                                currentBuild.result = 'UNSTABLE'
                                            } else {
                                                echo "Qualification tests passed!"
                                            }
                                        }
                                    }
                                }
                                
                                stage('Qualification Genre Tests') {
                                    when {
                                        expression { return env.TEST_TYPE == 'QUALIFICATION' }
                                    }
                                    steps {
                                        script {
                                            sh 'mkdir -p target/test-results'
                                            
                                            echo "=========================================="
                                            echo "Running QUALIFICATION GENRE tests"
                                            echo "=========================================="
                                            
                                            def testResult = sh(
                                                script: """
                                                    set +e
                                                    cargo test --test qualification_genre_tests -- --nocapture 2>&1 | tee target/test-results/qualification_genre_output.txt
                                                    exit \${PIPESTATUS[0]}
                                                """,
                                                returnStatus: true
                                            )
                                            
                                            if (testResult != 0) {
                                                echo "Qualification genre tests completed with failures"
                                                currentBuild.result = 'UNSTABLE'
                                            } else {
                                                echo "Qualification genre tests passed!"
                                            }
                                        }
                                    }
                                }
                                
                                stage('Regression Tests') {
                                    when {
                                        expression { return env.TEST_TYPE == 'REGRESSION' }
                                    }
                                    steps {
                                        script {
                                            sh 'mkdir -p target/test-results'
                                            
                                            echo "=========================================="
                                            echo "Running REGRESSION tests"
                                            echo "=========================================="
                                            
                                            def testResult = sh(
                                                script: """
                                                    set +e
                                                    cargo test --test regression_test -- --nocapture 2>&1 | tee target/test-results/regression_output.txt
                                                    exit \${PIPESTATUS[0]}
                                                """,
                                                returnStatus: true
                                            )
                                            
                                            if (testResult != 0) {
                                                echo "Regression tests completed with failures"
                                                currentBuild.result = 'UNSTABLE'
                                            } else {
                                                echo "Regression tests passed!"
                                            }
                                        }
                                    }
                                }
                            }
                        }
                        
                        stage('Regression Genre Tests') {
                            when {
                                expression { 
                                    return env.TEST_TYPE == 'REGRESSION_GENRE' || params.RUN_GENRE_REGRESSION == true
                                }
                            }
                            steps {
                                script {
                                    sh 'mkdir -p target/test-results'
                                    
                                    echo "=========================================="
                                    echo "Running REGRESSION GENRE tests (MANUAL ONLY)"
                                    echo "=========================================="
                                    
                                    def testResult = sh(
                                        script: """
                                            set +e
                                            cargo test --test regression_genre_tests -- --nocapture 2>&1 | tee target/test-results/regression_genre_output.txt
                                            exit \${PIPESTATUS[0]}
                                        """,
                                        returnStatus: true
                                    )
                                    
                                    if (testResult != 0) {
                                        echo "Regression genre tests completed with failures"
                                        currentBuild.result = 'UNSTABLE'
                                    } else {
                                        echo "Regression genre tests passed!"
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
        }
        unstable {
            echo 'Build completed but some tests failed. Check test results.'
        }
        failure {
            echo 'Build or tests failed. Check logs for details.'
        }
        always {
            // Archive the binary (on success or unstable)
            script {
                if (currentBuild.result != 'FAILURE') {
                    archiveArtifacts artifacts: 'target/release/audiocheckr', fingerprint: true, allowEmptyArchive: true
                }
            }
            
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
                    rm -f CompactTestFiles.zip TestFiles.zip GenreTestSuiteLite.zip TestSuite.zip
                    rm -rf CompactTestFiles TestFiles GenreTestSuiteLite TestSuite
                    
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
