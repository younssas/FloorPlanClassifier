import streamlit as st
import json
import os
import io
from typing import List, Dict, Any, Optional
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import Flow
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
import tempfile

class GoogleDriveManager:
    """Manages Google Drive authentication and file operations"""
    
    def __init__(self):
        self.SCOPES = ['https://www.googleapis.com/auth/drive.readonly']
        self.credentials_file = "google_drive_credentials.json"
        self.token_file = "google_drive_token.json"
        self.service = None
    
    def setup_credentials(self, client_id: str, client_secret: str, redirect_uri: str = "http://localhost:8080"):
        """Setup OAuth2 credentials for Google Drive access"""
        credentials_data = {
            "web": {
                "client_id": client_id,
                "client_secret": client_secret,
                "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                "token_uri": "https://oauth2.googleapis.com/token",
                "redirect_uris": [redirect_uri]
            }
        }
        
        with open(self.credentials_file, 'w') as f:
            json.dump(credentials_data, f, indent=2)
        
        return True
    
    def get_authorization_url(self) -> Optional[str]:
        """Get authorization URL for OAuth2 flow"""
        try:
            if not os.path.exists(self.credentials_file):
                return None
            
            flow = Flow.from_client_secrets_file(
                self.credentials_file, 
                scopes=self.SCOPES,
                redirect_uri="urn:ietf:wg:oauth:2.0:oob"  # For manual copy-paste flow
            )
            
            auth_url, _ = flow.authorization_url(prompt='consent')
            
            # Store flow in session state for later use
            st.session_state.oauth_flow = flow
            
            return auth_url
        except Exception as e:
            st.error(f"Error generating authorization URL: {str(e)}")
            return None
    
    def complete_authorization(self, auth_code: str) -> bool:
        """Complete OAuth2 authorization with the provided code"""
        try:
            if 'oauth_flow' not in st.session_state:
                st.error("Authorization flow not initialized")
                return False
            
            flow = st.session_state.oauth_flow
            flow.fetch_token(code=auth_code)
            
            # Save credentials
            with open(self.token_file, 'w') as f:
                f.write(flow.credentials.to_json())
            
            return True
        except Exception as e:
            st.error(f"Error completing authorization: {str(e)}")
            return False
    
    def authenticate(self) -> bool:
        """Authenticate with Google Drive using saved credentials"""
        try:
            creds = None
            
            # Load existing token
            if os.path.exists(self.token_file):
                creds = Credentials.from_authorized_user_file(self.token_file, self.SCOPES)
            
            # If credentials are not valid, return False
            if not creds or not creds.valid:
                if creds and creds.expired and creds.refresh_token:
                    try:
                        creds.refresh(Request())
                        # Save refreshed credentials
                        with open(self.token_file, 'w') as f:
                            f.write(creds.to_json())
                    except Exception:
                        return False
                else:
                    return False
            
            # Build the service
            self.service = build('drive', 'v3', credentials=creds)
            return True
            
        except Exception as e:
            st.error(f"Authentication error: {str(e)}")
            return False
    
    def list_files(self, folder_id: str = None, file_type: str = None) -> List[Dict[str, Any]]:
        """List files in Google Drive"""
        try:
            if not self.service:
                if not self.authenticate():
                    return []
            
            query = "trashed=false"
            
            if folder_id:
                query += f" and '{folder_id}' in parents"
            
            if file_type:
                if file_type.lower() == 'pdf':
                    query += " and mimeType='application/pdf'"
                elif file_type.lower() == 'json':
                    query += " and mimeType='application/json'"
                elif file_type.lower() == 'folder':
                    query += " and mimeType='application/vnd.google-apps.folder'"
            
            results = self.service.files().list(
                q=query,
                pageSize=100,
                fields="nextPageToken, files(id, name, mimeType, size, modifiedTime, parents)"
            ).execute()
            
            return results.get('files', [])
            
        except Exception as e:
            st.error(f"Error listing files: {str(e)}")
            return []
    
    def download_file(self, file_id: str, file_name: str) -> Optional[bytes]:
        """Download a file from Google Drive"""
        try:
            if not self.service:
                if not self.authenticate():
                    return None
            
            request = self.service.files().get_media(fileId=file_id)
            file_io = io.BytesIO()
            downloader = MediaIoBaseDownload(file_io, request)
            
            done = False
            while done is False:
                status, done = downloader.next_chunk()
            
            file_io.seek(0)
            return file_io.read()
            
        except Exception as e:
            st.error(f"Error downloading file {file_name}: {str(e)}")
            return None
    
    def search_files(self, search_term: str, file_type: str = None) -> List[Dict[str, Any]]:
        """Search for files by name"""
        try:
            if not self.service:
                if not self.authenticate():
                    return []
            
            query = f"name contains '{search_term}' and trashed=false"
            
            if file_type:
                if file_type.lower() == 'pdf':
                    query += " and mimeType='application/pdf'"
                elif file_type.lower() == 'json':
                    query += " and mimeType='application/json'"
            
            results = self.service.files().list(
                q=query,
                pageSize=50,
                fields="files(id, name, mimeType, size, modifiedTime)"
            ).execute()
            
            return results.get('files', [])
            
        except Exception as e:
            st.error(f"Error searching files: {str(e)}")
            return []

def render_google_drive_interface():
    """Render the Google Drive integration interface"""
    st.subheader("Google Drive Integration")
    
    # Initialize drive manager
    if 'drive_manager' not in st.session_state:
        st.session_state.drive_manager = GoogleDriveManager()
    
    drive_manager = st.session_state.drive_manager
    
    # Check if already authenticated
    if drive_manager.authenticate():
        st.success("Connected to Google Drive")
        
        # File browser interface
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.write("**Browse Your Files:**")
            
            # Search functionality
            search_term = st.text_input("Search files:", placeholder="Enter filename or keyword")
            file_type_filter = st.selectbox("File type:", ["All", "PDF", "JSON", "Folders"])
            
            if st.button("Search Files") or search_term:
                if search_term:
                    filter_type = file_type_filter.lower() if file_type_filter != "All" else None
                    if filter_type == "folders":
                        filter_type = "folder"
                    
                    files = drive_manager.search_files(search_term, filter_type)
                else:
                    files = drive_manager.list_files(file_type=file_type_filter.lower() if file_type_filter != "All" else None)
                
                if files:
                    st.write(f"Found {len(files)} files:")
                    
                    # Display files in a table format
                    for i, file in enumerate(files):
                        with st.expander(f"{file['name']} ({file.get('mimeType', 'unknown')})", expanded=False):
                            col_a, col_b = st.columns([2, 1])
                            
                            with col_a:
                                st.write(f"**ID:** {file['id']}")
                                st.write(f"**Size:** {file.get('size', 'N/A')} bytes")
                                st.write(f"**Modified:** {file.get('modifiedTime', 'N/A')}")
                            
                            with col_b:
                                if file['mimeType'] == 'application/pdf':
                                    if st.button(f"Use for Training", key=f"pdf_{i}"):
                                        # Download and use for training
                                        file_data = drive_manager.download_file(file['id'], file['name'])
                                        if file_data:
                                            st.session_state.drive_pdf_data = file_data
                                            st.session_state.drive_pdf_name = file['name']
                                            st.success(f"Loaded {file['name']} for training!")
                                
                                elif file['mimeType'] == 'application/json':
                                    if st.button(f"Load JSON", key=f"json_{i}"):
                                        # Download and use JSON data
                                        file_data = drive_manager.download_file(file['id'], file['name'])
                                        if file_data:
                                            try:
                                                json_data = json.loads(file_data.decode('utf-8'))
                                                st.session_state.drive_json_data = json_data
                                                st.session_state.drive_json_name = file['name']
                                                st.success(f"Loaded {file['name']} JSON data!")
                                            except Exception as e:
                                                st.error(f"Error parsing JSON: {str(e)}")
                else:
                    st.info("No files found matching your criteria")
        
        with col2:
            st.write("**Loaded Files:**")
            
            # Show loaded PDF
            if 'drive_pdf_data' in st.session_state:
                st.success(f"PDF: {st.session_state.drive_pdf_name}")
                if st.button("Use PDF for Training"):
                    return 'pdf', st.session_state.drive_pdf_data
            
            # Show loaded JSON
            if 'drive_json_data' in st.session_state:
                st.success(f"JSON: {st.session_state.drive_json_name}")
                if st.button("Use JSON for Analysis"):
                    return 'json', st.session_state.drive_json_data
    
    else:
        # Authentication setup
        st.warning("Google Drive access not configured")
        
        with st.expander("Setup Google Drive Access", expanded=True):
            st.markdown("""
            **To access your Google Drive files, you need to:**
            
            1. **Create Google Cloud Project & OAuth Credentials:**
               - Go to [Google Cloud Console](https://console.cloud.google.com/)
               - Create a new project or select existing one
               - Enable the Google Drive API
               - Create OAuth 2.0 credentials (Desktop application type)
               - Download the credentials JSON file
            
            2. **Enter your OAuth credentials below:**
            """)
            
            client_id = st.text_input("Client ID:", type="password")
            client_secret = st.text_input("Client Secret:", type="password")
            
            if st.button("Setup Credentials"):
                if client_id and client_secret:
                    if drive_manager.setup_credentials(client_id, client_secret):
                        st.success("Credentials saved!")
                        
                        # Generate authorization URL
                        auth_url = drive_manager.get_authorization_url()
                        if auth_url:
                            st.markdown(f"""
                            **Next Steps:**
                            1. Click this link to authorize access: [Authorize Google Drive]({auth_url})
                            2. Copy the authorization code from the redirect page
                            3. Paste it below to complete setup
                            """)
                            
                            auth_code = st.text_input("Authorization Code:")
                            if st.button("Complete Authorization"):
                                if auth_code:
                                    if drive_manager.complete_authorization(auth_code):
                                        st.success("Google Drive access authorized! Refresh the page.")
                                        st.rerun()
                                    else:
                                        st.error("Authorization failed. Please try again.")
                                else:
                                    st.error("Please enter the authorization code")
                else:
                    st.error("Please enter both Client ID and Client Secret")
    
    return None, None