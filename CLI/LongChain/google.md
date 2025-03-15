# Setting Up Google Cloud Service Account

To use Google Cloud services, you need to set up a service account and download its credentials in JSON format. Follow these steps:

1. **Create a Service Account**:
   - Go to the [Google Cloud Console](https://console.cloud.google.com/).
   - Navigate to **IAM & Admin** > **Service Accounts**.
   - Click on **Create Service Account**, fill in the details, and grant the necessary roles.

2. **Download the JSON Key**:
   - After creating the service account, select it and go to the **Keys** section.
   - Click **Add Key** > **Create new key**, then select **JSON**.
   - Download the JSON file and store it securely.

3. **Set the Environment Variable**:
   - On Windows, set the `GOOGLE_APPLICATION_CREDENTIALS` environment variable to the path of the JSON file:
     - **Command Prompt**:
       ```
       set GOOGLE_APPLICATION_CREDENTIALS=C:\path\to\your\service-account-file.json
       ```
     - **PowerShell**:
       ```
       $env:GOOGLE_APPLICATION_CREDENTIALS = "C:\path\to\your\service-account-file.json"
       ```

4. **Keep the File Secure**:
   - Do not share the JSON file publicly or commit it to version control systems.

This setup allows your application to authenticate with Google Cloud services.
