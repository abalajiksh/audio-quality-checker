pipeline {
    agent any
    
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
        pollSCM('H 2 * * 1')
    }
    
    stages {
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
                    
                    if (currentBuild.getBuildCauses('hudson.triggers.SCMTrigger$SCMTriggerCause')) {
                        env.TEST_TYPE = 'REGRESSION'
                    } else {
                        env.TEST_TYPE = 'QUALIFICATION'
                    }
                    echo "Test type: ${env.TEST_TYPE}"
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
                        
                        sh """
                            set -e
                            mc alias set myminio ${MINIO_ENDPOINT} ${MINIO_ACCESS_KEY} ${MINIO_SECRET_KEY}
                            
                            echo "=========================================="
                            echo "Downloading ${zipFile}"
                            echo "=========================================="
                            mc cp myminio/${MINIO_BUCKET}/${zipFile} .
                            
                            echo "Extracting test files..."
                            unzip -q -o ${zipFile}
                            
                            # Rename CompactTestFiles to TestFiles if needed
                            if [ -d "CompactTestFiles" ]; then
                                mv CompactTestFiles TestFiles
                            fi
                            
                            echo "Test files ready:"
                            find TestFiles -type f -name "*.flac" | wc -l
                            du -sh TestFiles
                        """
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
                                    } catch (Exception e) {
                                        echo "⚠️ SonarQube analysis failed: ${e.message}"
                                    }
                                }
                            }
                        }
                        
                        stage('Quality Gate') {
                            steps {
                                script {
                                    try {
                                        timeout(time: 5, unit: 'MINUTES') {
                                            def qg = waitForQualityGate abortPipeline: false
                                            if (qg.status != 'OK') {
                                                echo "⚠️ Quality Gate: ${qg.status}"
                                            } else {
                                                echo "✅ Quality Gate: PASSED"
                                            }
                                        }
                                    } catch (Exception e) {
                                        echo "⚠️ Quality Gate skipped: ${e.message}"
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
                                    if (env.TEST_TYPE == 'QUALIFICATION') {
                                        sh '''
                                            echo "=========================================="
                                            echo "Running QUALIFICATION tests"
                                            echo "=========================================="
                                            cargo test --test qualification_test -- --nocapture
                                        '''
                                    } else {
                                        // Check if regression is needed
                                        def significantChange = sh(
                                            script: 'git diff --name-only HEAD~1 HEAD | grep -E "^(src/|tests/)" || echo "none"',
                                            returnStdout: true
                                        ).trim()
                                        
                                        if (significantChange == "none") {
                                            echo "No significant changes. Skipping regression tests."
                                        } else {
                                            sh '''
                                                echo "=========================================="
                                                echo "Running REGRESSION tests"
                                                echo "=========================================="
                                                cargo test --test regression_test -- --nocapture
                                            '''
                                        }
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
            echo '✅ Build and tests completed successfully!'
            archiveArtifacts artifacts: 'target/release/audiocheckr', fingerprint: true
        }
        failure {
            echo '❌ Build or tests failed. Check logs for details.'
        }
        always {
            sh '''
                rm -f CompactTestFiles.zip TestFiles.zip
                rm -rf CompactTestFiles TestFiles
                echo "Workspace cleaned"
            '''
            junit allowEmptyResults: true, testResults: 'target/**/test-results/*.xml'
        }
    }
}