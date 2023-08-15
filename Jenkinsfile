pipeline {
    agent {
        label 'master'
    }

    stages {
        stage('Build docs using docker') {
            agent {
                label 'linux2'
            }
            
            steps {
                git changelog: false, \
                    credentialsId: 'GitHub-DIAGNijmegen-account', \
                    url: 'git@github.com:DIAGNijmegen/msk-tiger.git'
                
                sh script: 'docker build --tag tiger-docs --target docs_builder .'
            }
        }
        stage('Deploy docs') {
            agent {
                docker {
                    label 'linux2'
                    image 'tiger-docs'
                    args '--entrypoint="" '
                }
            }
            
            steps {
                sh script: 'rm -rf _docs && cp -a /tiger/docs/build/. _docs/'
            
                sshPublisher(
                    publishers: [
                        sshPublisherDesc(
                            configName: 'doc@repos.diagnijmegen.nl', 
                            transfers: [
                                sshTransfer(
                                    cleanRemote: true, 
                                    excludes: '', 
                                    execCommand: '', 
                                    execTimeout: 120000, 
                                    flatten: false, 
                                    makeEmptyDirs: false, 
                                    noDefaultExcludes: false, 
                                    patternSeparator: '[, ]+', 
                                    remoteDirectory: 'tiger', 
                                    remoteDirectorySDF: false, 
                                    removePrefix: '_docs/', 
                                    sourceFiles: '_docs/**'
                                )
                            ], 
                            usePromotionTimestamp: false, 
                            useWorkspaceInPromotion: false, 
                            verbose: true
                        )
                    ]
                )
            }
        }
    }
}
