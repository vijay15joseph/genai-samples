
This will start a web server on port 7860 that will serve the chatbot. You can then access the chatbot by visiting http://localhost:7860 in your web browser.

## Setup
Enabling CloudRun with access to Vertex AI and Gemini APIs in Google Cloud Project
**Cloud Shell commands to create Service Account and provide IAM access for CloudRUn to access Vertex AI and Gemini APIs** 
    # Set the environment vars based on your environment
    PROJECT_ID=<YOUR_PROJECT_ID>
    EMAIL=<YOUR_USER_ACCOUNT>

    # Enable Cloud Run API
    gcloud services enable run.googleapis.com

    # Create service account
    gcloud iam service-accounts create cloud-run-gemini \
        --description="Service account to call gemini models from Cloud Run" \
        --display-name="cloud-run-gemini"

    # add aiplatform.user role
    gcloud projects add-iam-policy-binding ${PROJECT_ID} \
        --member="serviceAccount:cloud-run-gemini@${PROJECT_ID}.iam.gserviceaccount.com" \
        --role="roles/aiplatform.user"

    # add logging.logWriter role
    gcloud projects add-iam-policy-binding ${PROJECT_ID} \
        --member="serviceAccount:cloud-run-gemini@${PROJECT_ID}.iam.gserviceaccount.com" \
        --role="roles/logging.logWriter"

    # add permission to impersonate the sa (iam.serviceAccounts.actAs), since this is a user-namaged sa
    gcloud iam service-accounts add-iam-policy-binding \
        cloud-run-gemini@${PROJECT_ID}.iam.gserviceaccount.com \
        --member="user:${EMAIL}" \
        --role="roles/iam.serviceAccountUser"

**Build and deploy CloudRun** 
    PROJECT_ID=<REPLACE_WITH_YOUR_PROJECT_ID>
    REGION=<REPLACE_WITH_YOUR_GCP_REGION_NAME>
    AR_REPO=<REPLACE_WITH_YOUR_AR_REPO_NAME>
    SERVICE_NAME=genai-text-demo
    gcloud artifacts repositories create $AR_REPO --location=$REGION --repository-format=Docker
    gcloud auth configure-docker $REGION-docker.pkg.dev
    gcloud builds submit --tag $REGION-docker.pkg.dev/$PROJECT_ID/$AR_REPO/$SERVICE_NAME
    gcloud run deploy $SERVICE_NAME --port 7860 --image $REGION-docker.pkg.dev/$PROJECT_ID/$AR_REPO/$SERVICE_NAME --service-account=cloud-run-llm@$PROJECT_ID.iam.gserviceaccount.com --allow-unauthenticated --region=$REGION --platform=managed  --project=$PROJECT_ID

## Example

The following example shows how to use the chatbot to generate text and images:

1. Visit http://localhost:7860 in your web browser.
2. Type a prompt into the text box at the bottom of the page.
3. Click the "Submit" button.

The chatbot will generate text and images based on your prompt. The text will be displayed in the chat window, and the images will be displayed in the gallery below the chat window.

## Troubleshooting

If you encounter any problems using this image, please refer to the following troubleshooting tips:

* Make sure that you have Docker installed and running on your system.
* Make sure that you have pulled the latest version of the image from Docker Hub.
* Make sure that you are running the image with the correct port mapping.
* Make sure that you have allowed access to port 7860 in your firewall.

If you are still having problems, please feel free to create an issue on the GitHub repository for this project.
