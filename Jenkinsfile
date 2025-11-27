pipeline {
    agent any
    
    environment {
        // MinIO configuration - will be loaded from Jenkins credentials
        MINIO_BUCKET = 'audiocheckr'
        MINIO_FILE = 'TestFiles.zip'
        
        // SonarQube configuration
        SONAR_PROJECT_KEY = 'audiocheckr'
        SONAR_PROJECT_NAME = 'AudioCheckr'
        SONAR_SOURCES = 'src'
        
        // Add user bin and cargo to PATH
        PATH = "$HOME/bin:$HOME/.cargo/bin:/usr/bin:$PATH"
    }
    
    triggers {
        // Weekly regression test - Mondays at 2 AM, only if changes exist
        pollSCM('H 2 * * 1')
    }
    
    stages {
        stage('Setup Tools') {
    steps {
        sh '''
            # Create user bin directory if it doesn't exist
            mkdir -p $HOME/bin
            
            # Verify build tools are installed (POSIX-compliant check)
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
                }
            }
        }
        
        stage('Download Test Files from MinIO') {
            steps {
                script {
                    // Load MinIO credentials from Jenkins
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
                            # Configure MinIO client
                            mc alias set myminio ${MINIO_ENDPOINT} ${MINIO_ACCESS_KEY} ${MINIO_SECRET_KEY}
                            
                            # Download test files
                            echo "Downloading ${MINIO_FILE} from MinIO..."
                            mc cp myminio/${MINIO_BUCKET}/${MINIO_FILE} .
                            
                            # Extract to project root
                            echo "Extracting test files..."
                            unzip -q -o ${MINIO_FILE}
                            
                            # Verify extraction
                            ls -lh TestFiles/ | head -n 10
                        '''
                    }
                }
            }
        }
        
        stage('Build') {
            steps {
                sh '''
                    echo "Building Rust project..."
                    
                    # Build release binary
                    cargo build --release
                    
                    # Verify binary exists and show info
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
                    // SonarQube scanner tool configured in Jenkins Global Tool Configuration
                    def scannerHome = tool 'SonarQube-LXC'
                    
                    // withSonarQubeEnv uses the SonarQube server configured in Jenkins System Configuration
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
        
        stage('Validation Test - Quick') {
            when {
                anyOf {
                    triggeredBy 'GitHubPushCause'
                    triggeredBy 'UserIdCause'
                    branch 'main'
                }
            }
            steps {
                sh '''
                    echo "=========================================="
                    echo "Running validation tests (22 files)..."
                    echo "=========================================="
                    cargo test --test validation_test -- --nocapture
                '''
            }
        }
        
        stage('Determine Regression Necessity') {
            when {
                triggeredBy 'SCMTrigger'
            }
            steps {
                script {
                    // Check if changes are significant
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
        
        stage('Regression Test - Full') {
            when {
                allOf {
                    triggeredBy 'SCMTrigger'
                    environment name: 'RUN_REGRESSION', value: 'true'
                }
            }
            steps {
                sh '''
                    echo "=========================================="
                    echo "Running full regression tests (82 files)..."
                    echo "=========================================="
                    cargo test --test regression_test_ground_truth -- --nocapture
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
            // Clean up test files to save space
            sh '''
                rm -f TestFiles.zip
                echo "Workspace cleaned"
            '''
            
            // Publish test results if available
            junit allowEmptyResults: true, testResults: 'target/**/test-results/*.xml'
        }
    }
}
