pipeline {
    agent any
    
    environment {
        // MinIO configuration - using Jenkins global credentials
        MINIO_ENDPOINT = 'http://your-minio-lxc-ip:9000'
        MINIO_BUCKET = 'audiocheckr'
        MINIO_FILE = 'TestFiles.zip'
        
        // SonarQube configuration
        SONAR_PROJECT_KEY = 'audiocheckr'
        SONAR_PROJECT_NAME = 'AudioCheckr'
        SONAR_SOURCES = 'src'
    }
    
    triggers {
        // Weekly regression test - Mondays at 2 AM, only if changes exist
        pollSCM('H 2 * * 1')
    }
    
    stages {
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
                    // Use the MinIO credentials stored in Jenkins
                    withCredentials([usernamePassword(
                        credentialsId: 'noIdea',
                        usernameVariable: 'MINIO_ACCESS_KEY',
                        passwordVariable: 'MINIO_SECRET_KEY'
                    )]) {
                        sh '''
                            # Install mc (MinIO client) if not present
                            if ! command -v mc &> /dev/null; then
                                wget https://dl.min.io/client/mc/release/linux-amd64/mc
                                chmod +x mc
                                sudo mv mc /usr/local/bin/
                            fi
                            
                            # Configure MinIO client
                            mc alias set myminio ${MINIO_ENDPOINT} ${MINIO_ACCESS_KEY} ${MINIO_SECRET_KEY}
                            
                            # Download test files
                            mc cp myminio/${MINIO_BUCKET}/${MINIO_FILE} .
                            
                            # Extract to project root
                            unzip -o ${MINIO_FILE}
                        '''
                    }
                }
            }
        }
        
        stage('Build') {
            steps {
                sh '''
                    # Ensure Rust is in PATH
                    export PATH="$HOME/.cargo/bin:$PATH"
                    
                    # Build release binary
                    cargo build --release
                    
                    # Verify binary exists
                    ls -lh target/release/audiocheckr
                '''
            }
        }
        
        stage('SonarQube Analysis') {
            steps {
                script {
                    // SonarQube scanner tool configured in Jenkins Global Tool Configuration
                    def scannerHome = tool 'SonarQubeScanner'
                    
                    // withSonarQubeEnv uses the SonarQube server configured in Jenkins System Configuration
                    // and automatically uses the token stored in Jenkins credentials
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
                    // Uses SonarQube webhook configured in SonarQube server
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
                    # Ensure Rust is in PATH
                    export PATH="$HOME/.cargo/bin:$PATH"
                    
                    echo "Running validation tests (22 representative files)..."
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
                    # Ensure Rust is in PATH
                    export PATH="$HOME/.cargo/bin:$PATH"
                    
                    echo "Running full regression tests (82 files with ground truth)..."
                    cargo test --test regression_test_ground_truth -- --nocapture
                '''
            }
        }
    }
    
    post {
        success {
            echo 'Build and tests completed successfully!'
            archiveArtifacts artifacts: 'target/release/audiocheckr', fingerprint: true
        }
        failure {
            echo 'Build or tests failed. Check logs for details.'
        }
        always {
            // Clean up test files to save space
            sh 'rm -f TestFiles.zip'
            
            // Publish test results if available
            junit allowEmptyResults: true, testResults: 'target/**/test-results/*.xml'
        }
    }
}
