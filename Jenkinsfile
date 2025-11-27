pipeline {
    agent any
    
    environment {
        // MinIO configuration
        MINIO_BUCKET = 'audiocheckr'
        // Use CompactTestFiles.zip for qualification, TestFiles.zip for regression
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
        // Weekly regression test - Mondays at 2 AM
        pollSCM('H 2 * * 1')
    }
    
    stages {
        stage('Setup Tools') {
            steps {
                sh '''
                    # Create user bin directory if it doesn't exist
                    mkdir -p $HOME/bin
                    
                    # Verify build tools are installed
                    if ! command -v cc >/dev/null 2>&1; then
                        echo "ERROR: C compiler not found!"
                        echo "Please run on Jenkins server:"
                        echo "  apt update && apt install -y build-essential pkg-config libssl-dev"
                        exit 1
                    fi
                    
                    # Install MinIO client if not present
                    if ! command -v mc >/dev/null 2>&1; then
                        echo "Installing MinIO client..."
                        wget -q https://dl.min.io/client/mc/release/linux-amd64/mc -O $HOME/bin/mc
                        chmod +x $HOME/bin/mc
                    fi
                    
                    # Install Rust if not present
                    if ! command -v cargo >/dev/null 2>&1; then
                        echo "Installing Rust..."
                        curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
                        . $HOME/.cargo/env
                    fi
                    
                    # Verify installations
                    echo "=== Tool Versions ==="
                    mc --version
                    cargo --version
                    rustc --version
                    cc --version
                    echo "===================="
                '''
            }
        }
        
        stage('Checkout') {
            steps {
                checkout scm
                script {
                    env.GIT_COMMIT_MSG = sh(
                        script: 'git log -1 --pretty=%B',
                        returnStdout: true
                    ).trim()
                    env.CHANGED_FILES = sh(
                        script: 'git diff --name-only HEAD~1 HEAD | wc -l',
                        returnStdout: true
                    ).trim()
                    
                    // Determine test type based on trigger
                    if (currentBuild.getBuildCauses('hudson.triggers.SCMTrigger$SCMTriggerCause')) {
                        env.TEST_TYPE = 'REGRESSION'
                    } else {
                        env.TEST_TYPE = 'QUALIFICATION'
                    }
                    echo "Test type: ${env.TEST_TYPE}"
                }
            }
        }
        
        stage('Download Compact Test Files') {
            when {
                environment name: 'TEST_TYPE', value: 'QUALIFICATION'
            }
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
                        sh '''
                            set -e
                            
                            # Configure MinIO client
                            mc alias set myminio ${MINIO_ENDPOINT} ${MINIO_ACCESS_KEY} ${MINIO_SECRET_KEY}
                            
                            # Download compact test files
                            echo "=========================================="
                            echo "Downloading COMPACT test files"
                            echo "=========================================="
                            mc cp myminio/${MINIO_BUCKET}/${MINIO_FILE_COMPACT} .
                            
                            # Extract (Windows ZIPs may have backslash warnings - ignore them)
                            echo "Extracting test files..."
                            unzip -q -o ${MINIO_FILE_COMPACT}
                            
                            # Show what was extracted
                            echo "Extracted contents:"
                            ls -la
                            
                            # The ZIP might create CompactTestFiles/ - rename to TestFiles for consistency
                            if [ -d "CompactTestFiles" ]; then
                                echo "Found CompactTestFiles/, renaming to TestFiles/"
                                mv CompactTestFiles TestFiles
                            fi
                            
                            # Verify TestFiles exists
                            if [ -d "TestFiles" ]; then
                                echo "Test files ready:"
                                find TestFiles -type f -name "*.flac" | wc -l
                                du -sh TestFiles
                            else
                                echo "ERROR: TestFiles directory not found!"
                                ls -la
                                exit 1
                            fi
                        '''
                    }
                }
            }
        }
        
        stage('Download Full Test Files') {
            when {
                environment name: 'TEST_TYPE', value: 'REGRESSION'
            }
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
                        sh '''
                            set -e
                            
                            # Configure MinIO client
                            mc alias set myminio ${MINIO_ENDPOINT} ${MINIO_ACCESS_KEY} ${MINIO_SECRET_KEY}
                            
                            # Download full test files
                            echo "=========================================="
                            echo "Downloading FULL test files"
                            echo "=========================================="
                            mc cp myminio/${MINIO_BUCKET}/${MINIO_FILE_FULL} .
                            
                            # Extract
                            echo "Extracting test files..."
                            unzip -q -o ${MINIO_FILE_FULL}
                            
                            # Verify
                            echo "Test files extracted:"
                            if [ -d "TestFiles" ]; then
                                find TestFiles -type f -name "*.flac" | wc -l
                                du -sh TestFiles
                            else
                                echo "ERROR: TestFiles directory not found!"
                                ls -la
                                exit 1
                            fi
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
                    file target/release/audiocheckr
                    echo "======================"
                '''
            }
        }
        
        stage('SonarQube Analysis') {
            steps {
                script {
                    def scannerHome = tool 'SonarQube-LXC'
                    
                    withSonarQubeEnv('slxc') {
                        sh """
                            ${scannerHome}/bin/sonar-scanner \
                                -Dsonar.projectKey=${SONAR_PROJECT_KEY} \
                                -Dsonar.projectName=${SONAR_PROJECT_NAME} \
                                -Dsonar.sources=${SONAR_SOURCES} \
                                -Dsonar.rust.clippy.reportPaths=target/clippy-report.json \
                                -Dsonar.exclusions=**/target/**,**/TestFiles/**
                        """
                    }
                }
            }
        }
        
        stage('Quality Gate') {
            steps {
                timeout(time: 5, unit: 'MINUTES') {
                    waitForQualityGate abortPipeline: true
                }
            }
        }
        
        stage('Qualification Test') {
            when {
                environment name: 'TEST_TYPE', value: 'QUALIFICATION'
            }
            steps {
                sh '''
                    echo "=========================================="
                    echo "Running QUALIFICATION tests (20 files)"
                    echo "=========================================="
                    cargo test --test qualification_test -- --nocapture
                '''
            }
        }
        
        stage('Determine Regression Necessity') {
            when {
                environment name: 'TEST_TYPE', value: 'REGRESSION'
            }
            steps {
                script {
                    def significantChange = sh(
                        script: '''
                            # Check if src/ or tests/ directories changed
                            git diff --name-only HEAD~1 HEAD | grep -E '^(src/|tests/)' || echo "none"
                        ''',
                        returnStdout: true
                    ).trim()
                    
                    if (significantChange == "none") {
                        echo "No significant changes detected (README/docs only). Skipping regression."
                        env.RUN_REGRESSION = "false"
                    } else {
                        echo "Significant changes detected: ${significantChange}"
                        env.RUN_REGRESSION = "true"
                    }
                }
            }
        }
        
        stage('Regression Test') {
            when {
                allOf {
                    environment name: 'TEST_TYPE', value: 'REGRESSION'
                    environment name: 'RUN_REGRESSION', value: 'true'
                }
            }
            steps {
                sh '''
                    echo "=========================================="
                    echo "Running REGRESSION tests (80+ files)"
                    echo "=========================================="
                    cargo test --test regression_test -- --nocapture
                '''
            }
        }
    }
    
    post {
        success {
            echo '✅ Build and tests completed successfully!'
            archiveArtifacts artifacts: 'target/release/audiocheckr', fingerprint: true
        }
        failure {
            echo '❌ Build or tests failed. Check logs for details.'
        }
        always {
            sh '''
                # Clean up test files to save space
                rm -f CompactTestFiles.zip TestFiles.zip
                rm -rf CompactTestFiles TestFiles
                echo "Workspace cleaned"
            '''
            
            junit allowEmptyResults: true, testResults: 'target/**/test-results/*.xml'
        }
    }
}