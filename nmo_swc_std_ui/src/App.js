/**
 * File: App.js
 * Author: Herve Emissah
 * Created: 2021-08-15
 * Description: Main React component for the NMO SWC QC web interface.
 *              Handles files folder selection, log viewing, and downloading of standardized files.
 */


import React, { useEffect, useState, useRef } from 'react';
import './App.css';
import logo from './nmo_swc_qc_logo.png';
import LogViewer from './LogViewer';

function App() {
  const [selectedFiles, setSelectedFiles] = useState(null);
  const [error, setError] = useState(null);
  const [logContent, setLogContent] = useState('');
  const [checkCorrectBranchTag, setCheckCorrectBranchTag] = useState(false);
  const [checkLongConnections, setCheckLongConnections] = useState(true);
  const [branchtype, setbranchtype] = useState(2); // Set default value to 2
  const [stdevX, setStdevX] = useState(6); // Set default value to 6
  const [isSaving, setIsSaving] = useState(false); // State for tracking saving status
  const [systemBusy, setSystemBusy] = useState(false);

  const branchSelectRef = useRef(null);
  const branchTextRef = useRef(null);

  const stdevSelectRef = useRef(null);
  const stdevTextRef = useRef(null);
  
  const branchLabels = {
    2: 'Axon (2)',
    3: 'Basal dendrites (3)',
    4: 'Apical dendrites (4)',
    5: 'Other dendrites (5)',
    6: 'Unspecified neurites (6)',
    7: 'Glial processes (7)',
  };

  useEffect(() => {
    if (branchSelectRef.current && branchTextRef.current) {
      branchSelectRef.current.style.width =
        branchTextRef.current.offsetWidth + 32 + 'px'; // arrow padding
    }
  }, [branchtype]);

  useEffect(() => {
    if (stdevSelectRef.current && stdevTextRef.current) {
      stdevSelectRef.current.style.width =
        stdevTextRef.current.offsetWidth + 32 + 'px'; // arrow padding
    }
  }, [stdevX]);

  const handleFileChange = (e) => {
    const files = e.target.files;
    setSelectedFiles(files);
  };

  const handleSubmit = async (e) => {
    e.preventDefault();

    if (!selectedFiles) {
      return;
    }

    setIsSaving(true); // Set saving state to true when saving starts

    const formData = new FormData();
    for (const file of selectedFiles) {
      formData.append('files', file);
    }

    try {
      // Send a POST request to your backend to handle file upload.
      const response = await fetch('/nmo/upload', {
        method: 'POST',
        body: formData,
      });

      if (response.status === 429) {
         setSystemBusy(true);
         window.alert('System currently in use. Please try again later.');
         return;
      }

      if (response.ok) {
        //window.alert('Upload to Server completed successfully.');
      } else {
        window.alert('Failed to upload files to server.');
      }

      // Handle the response (e.g., display a success message).
    } catch (error) {
      // Handle errors (e.g., display an error message).
      console.error('Error uploading files:', error);
    } finally {
      setIsSaving(false); // Set saving state to false when saving ends
      setSystemBusy(false);
    }

  };

  const handleStandardizeClick = async () => {
    try {
      const formData = new FormData();
      for (const file of selectedFiles) {
        formData.append('files', file);
      }
      formData.append('checkCorrectBranchTag', checkCorrectBranchTag);
      formData.append('branchtype', branchtype);

      const response = await fetch('/nmo/SWC_STD', {
        method: 'POST',
        body: formData,
      });
	  
      if (response.status === 429) {
         setSystemBusy(true);
         setIsSaving(false);
         window.alert('System currently in use. Please try again later.');
         return;
      }

      if (response.ok) {
        //window.alert('Standardize process completed successfully.');
        const shouldDownload = window.confirm(
           'Standardize process completed successfully.\nDo you want to download the files?'
        );

        if (shouldDownload) {
          await handleDownloadClick();
        }
        setSystemBusy(false);
      } else {
        window.alert('Failed to complete Standardize process.');
      }
    } catch (error) {
      console.error('Error during Standardize:', error);
      window.alert('Error occurred during Standardize process.');
    }
  };

  const handleAutoConnectClick = async () => {
    try {
      const formData = new FormData();
      for (const file of selectedFiles) {
        formData.append('files', file);
      }
      formData.append('checkLongConnections', checkLongConnections);
      formData.append('stdevX', stdevX);

      const response = await fetch('/nmo/connect_disjoint_subtrees', {
        method: 'POST',
        body: formData,
      });

      if (response.status === 429) {
         setSystemBusy(true);
         setIsSaving(false);
         window.alert('System currently in use. Please try again later.');
         return;
      }

      if (response.ok) {
        //window.alert('AutoConnect process completed successfully.');
        const shouldDownload = window.confirm(
           'AutoConnect process completed successfully.\nDo you want to download the files?'
        );

        if (shouldDownload) {
          await handleDownloadConnectClick ();
        }
        setSystemBusy(false);
      } else {
        window.alert('Failed to complete AutoConnect process.');
      }
    } catch (error) {
      console.error('Error during AutoConnect:', error);
      window.alert('Error occurred during AutoConnect process.');
    }
  };

  const handleAutoTagClick = async () => {
    try {
      const response = await fetch('/nmo/CorrectTag', {
        method: 'POST',
        body: selectedFiles,
      });

      if (response.status === 429) {
         setSystemBusy(true);
         setIsSaving(false);
         window.alert('System currently in use. Please try again later.');
         return;
      }

      if (response.ok) {
        //window.alert('Tag correction process completed successfully.');
        const shouldDownload = window.confirm(
           'Auto Tag process completed successfully.\nDo you want to download the files?'
        );

        if (shouldDownload) {
          await handleDownloadCorrectedTagClick();
        }
        setSystemBusy(false);
      } else {
        window.alert('Failed to complete Tag correction process.');
      }
    } catch (error) {
      console.error('Error during Tag Correction:', error);
      window.alert('Error occurred during Tag Correction process.');
    }
  };

  const handleDownloadClick = async () => {
    try {
      const timestamp = new Date().toLocaleString('en-US', {
  	month: 'numeric',
  	day: 'numeric',
  	year: 'numeric',
  	hour: 'numeric',
  	minute: 'numeric',
  	second: 'numeric',
  	hour12: false
      }).replace(/[/:,\s]/g, '_');
      const downloadUrl = `/nmo/download?timestamp=${timestamp}`;

      // Make a GET request to the Flask server's /nmo/download route
      //const response = await fetch('/nmo/download');
      const response = await fetch(downloadUrl);
      if (!response.ok) {
        throw new Error(`HTTP error! Status: ${response.status}`);
      }

      const blob = await response.blob();

      // Create a URL for the blob data and initiate the download
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `swc_standardized_${timestamp}.zip`;
      a.click();
      window.URL.revokeObjectURL(url);

      // Display a popup window after the download completes
      window.alert('Download completed successfully.');

    } catch (error) {
      console.error('Error downloading file:', error);
    }
  };

  const handleDownloadConnectClick = async () => {
    try {
      const timestamp = new Date().toLocaleString('en-US', {
        month: 'numeric',
        day: 'numeric',
        year: 'numeric',
        hour: 'numeric',
        minute: 'numeric',
        second: 'numeric',
        hour12: false
      }).replace(/[/:,\s]/g, '_');
      const downloadUrl = `/nmo/download_connected?timestamp=${timestamp}`;
      console.log('Download URL:', downloadUrl);

      // Make a GET request to the Flask server's /nmo/download route
      //const response = await fetch('/nmo/download');
      const response = await fetch(downloadUrl);
      console.log('Response status:', response.status);
      if (!response.ok) {
        throw new Error(`HTTP error! Status: ${response.status}`);
      }

      const blob = await response.blob();
      console.log('Blob size:', blob.size);

      // Create a URL for the blob data and initiate the download
      const url = window.URL.createObjectURL(blob);
      console.log('Blob URL:', url);
      const a = document.createElement('a');
      a.href = url;
      a.download = `swc_connected_${timestamp}.zip`;
      document.body.appendChild(a);
      console.log('Anchor tag appended to body');
      a.click();
      console.log('Download initiated');
      document.body.removeChild(a);
      window.URL.revokeObjectURL(url);

      // Display a popup window after the download completes
      window.alert('Download completed successfully.');
      console.log('Blob URL revoked and anchor tag removed');

    } catch (error) {
      console.error('Error downloading file:', error);
    }
  };

  const handleDownloadCorrectedTagClick = async () => {
    try {
      const timestamp = new Date().toLocaleString('en-US', {
        month: 'numeric',
        day: 'numeric',
        year: 'numeric',
        hour: 'numeric',
        minute: 'numeric',
        second: 'numeric',
        hour12: false
      }).replace(/[/:,\s]/g, '_');
      const downloadUrl = `/nmo/download_corrected_tags?timestamp=${timestamp}`;
      console.log('Download URL:', downloadUrl);

      // Make a GET request to the Flask server's /nmo/download route
      const response = await fetch(downloadUrl);
      console.log('Response status:', response.status);
      if (!response.ok) {
        throw new Error(`HTTP error! Status: ${response.status}`);
      }

      const blob = await response.blob();
      console.log('Blob size:', blob.size);

      // Create a URL for the blob data and initiate the download
      const url = window.URL.createObjectURL(blob);
      console.log('Blob URL:', url);
      const a = document.createElement('a');
      a.href = url;
      a.download = `swc_auto_tag_${timestamp}.zip`;
      document.body.appendChild(a);
      console.log('Anchor tag appended to body');
      a.click();
      console.log('Download initiated');
      document.body.removeChild(a);
      window.URL.revokeObjectURL(url);

      // Display a popup window after the download completes
      window.alert('Download completed successfully.');
      console.log('Blob URL revoked and anchor tag removed');

    } catch (error) {
      console.error('Error downloading file:', error);
    }
  };


const sharedGridColumns = '360px 260px';

const buttonStyle = {
  fontSize: '16px',
  padding: '6px 20px',
  backgroundColor: '#d3d3d3',
  color: 'green',
  border: '1px solid black',
  cursor: 'pointer',
  width: '140px',
  marginLeft: '5px',
  marginTop: '10px',
  marginBottom: '10px',
  marginRight: '20px',
  whiteSpace: 'nowrap',
  display: 'inline-flex',
  alignItems: 'center',
  justifyContent: 'center',
};

const formStyle = {
  display: 'flex',
  alignItems: 'center',
  gap: '20px',
  marginTop: '15px',
  marginLeft: '5px',
  marginBottom: '50px',
};

const row2GridStyle = {
  display: 'grid',
  gridTemplateColumns: sharedGridColumns,
  columnGap: '80px',
  alignItems: 'start',
  marginTop: '10px',
  marginBottom: '25px',
  marginLeft: '5px',
  position: 'relative',
};

const row3GridStyle = {
  display: 'grid',
  gridTemplateColumns: '360px 260px 260px',
  columnGap: '80px',
  marginBottom: '25px',
  marginLeft: '5px',
};

const checkboxLabelStyle = {
  display: 'flex',
  alignItems: 'center',
  gap: '4px',
};

const subRowStyle = {
  marginTop: '2px',
  marginLeft: '18px',
  display: 'flex',
  alignItems: 'center',
  gap: '6px',
  flexWrap: 'nowrap',
};

const selectStyle = {
  whiteSpace: 'nowrap',
  fontSize: '16px',
  fontFamily: 'inherit',
};


return (
  <div align="left">
    <img
      src={logo}
      width="700"
      height="100"
      alt="swc qc logo"
      style={{ marginBottom: '10px', marginLeft: '5px', marginTop: '5px' }}
    />

    {/* ---------- ROW 1: Upload ---------- */}
    <form onSubmit={handleSubmit} style={formStyle}>
      <input
        type="file"
        multiple
        webkitdirectory="true"
        accept=".swc"
        onChange={handleFileChange}
        style={{ color: 'green', width: '240px', fontSize: '16px' }}
      />

      <button type="submit" style={{ ...buttonStyle, marginLeft: '80px' }}>
        {isSaving ? 'Saving...' : 'Upload To Server'}
      </button>
    </form>

    {/* ---------- ROW 2: Options (GRID = stable) ---------- */}
    <div style={row2GridStyle}>
      {/* Column 1: Correct Branch Tag */}
      <div>
        <label style={checkboxLabelStyle}>
          <input
            type="checkbox"
            checked={checkCorrectBranchTag}
            onChange={(e) => setCheckCorrectBranchTag(e.target.checked)}
          />
          Correct Branch Tag
        </label>

        <div style={subRowStyle}>
          <label style={{ marginRight: '4px' }}>New BranchType</label>

          {/* Hidden width calculator for BranchType */}
          <span
            ref={branchTextRef}
            style={{
              position: 'absolute',
              visibility: 'hidden',
              whiteSpace: 'nowrap',
              fontSize: '16px',
              fontFamily: 'inherit',
              fontWeight: 'normal',
            }}
          >
            {branchLabels[branchtype]}
          </span>

          <select
            ref={branchSelectRef}
            value={branchtype}
            onChange={(e) => setbranchtype(Number(e.target.value))}
            style={selectStyle}
          >
            <option value={2}>Axon (2)</option>
            <option value={3}>Basal dendrites (3)</option>
            <option value={4}>Apical dendrites (4)</option>
            <option value={5}>Other dendrites (5)</option>
            <option value={6}>Unspecified neurites (6)</option>
            <option value={7}>Glial processes (7)</option>
          </select>
        </div>
      </div>

      {/* Column 2: Fix long connections */}
      <div style={{ marginLeft: '-110px' }}>
        <label style={checkboxLabelStyle}>
          <input
            type="checkbox"
            checked={checkLongConnections}
            onChange={(e) => setCheckLongConnections(e.target.checked)}
          />
          Fix long connections
        </label>

        <div style={subRowStyle}>
          <label style={{ marginRight: '6px' }}>Use Stdev X</label>

          {/* Hidden width calculator for StdevX */}
          <span
            ref={stdevTextRef}
            style={{
              position: 'absolute',
              visibility: 'hidden',
              whiteSpace: 'nowrap',
              fontSize: '16px',
              fontFamily: 'inherit',
              fontWeight: 'normal',
            }}
          >
            {String(stdevX)}
          </span>

          <select
            ref={stdevSelectRef}
            value={stdevX}
            onChange={(e) => setStdevX(Number(e.target.value))}
            style={selectStyle}
          >
            {Array.from({ length: 7 }, (_, i) => i + 4).map((num) => (
              <option key={num} value={num}>
                {num}
              </option>
            ))}
          </select>
        </div>
      </div>
    </div>

    {/* ---------- ROW 3: Actions (THREE columns) ---------- */}
    <div style={row3GridStyle}>
      {/* Column 1 */}
      <div>
        <button onClick={handleStandardizeClick} style={buttonStyle}>
          Standardize
        </button>
      </div>

      {/* Column 2: aligned under Fix long connections */}
      <div style={{ marginLeft: '-110px' }}>
        <button onClick={handleAutoConnectClick} style={buttonStyle}>
          Auto Connect
        </button>
      </div>

      {/* Column 3: Auto Tag Apical on third row */}
      <div  style={{ marginLeft: '-200px' }}>
        <button onClick={handleAutoTagClick} style={buttonStyle}>
          Auto Tag Apical
        </button>
      </div>
    </div>

    {/* ---------- Logs ---------- */}
    <LogViewer />
  </div>
);

}

export default App;

