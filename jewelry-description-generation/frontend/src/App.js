import React, { useState } from 'react';
import axios from 'axios';
import { Upload, Loader2, Scan, Gem, AlertCircle } from 'lucide-react';

function App() {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const handleFileChange = (e) => {
    const selectedFile = e.target.files[0];
    
    if (!selectedFile) return;

    // 1. VALIDATION: Check if file is an image
    if (!selectedFile.type.startsWith('image/')) {
      setError('Please upload a valid image file.');
      setFile(null);
      setPreview(null);
      return;
    }

    setFile(selectedFile);
    setPreview(URL.createObjectURL(selectedFile));
    setResult(null);
    setError('');
  };

  const handleAnalyze = async () => {
    if (!file) return;

    setLoading(true);
    setError('');
    
    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await axios.post('http://localhost:8000/analyze', formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
      });
      setResult(response.data);
    } catch (err) {
      console.error(err);
      setError('Analysis failed. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen flex items-center justify-center p-6 bg-stone-100 text-stone-800">
      <div className="max-w-5xl w-full bg-white rounded-xl shadow-xl overflow-hidden flex flex-col md:flex-row border border-stone-200">
        
        {/* Left Side - Upload Area */}
        <div className="w-full md:w-1/2 p-8 md:p-12 flex flex-col items-center justify-center bg-stone-50">
          <div className="mb-8 text-center">
            <h1 className="text-3xl font-serif font-bold text-emerald-900 mb-2">
              Jewellery Analyzer
            </h1>
            <p className="text-stone-500 text-sm tracking-wide">
              UPLOAD JEWELLERY FOR IDENTIFICATION
            </p>
          </div>

          <div className="w-full relative group">
            <input 
              type="file" 
              accept="image/*" 
              onChange={handleFileChange} 
              className="absolute inset-0 w-full h-full opacity-0 cursor-pointer z-10"
            />
            <div className={`
              border-2 border-dashed rounded-lg h-80 flex flex-col items-center justify-center transition-all duration-300 overflow-hidden bg-white
              ${error ? 'border-red-300 bg-red-50' : preview ? 'border-emerald-500' : 'border-stone-300 hover:border-emerald-400 hover:bg-stone-50'}
            `}>
              {preview ? (
                // CHANGED: object-contain prevents cropping
                <img 
                  src={preview} 
                  alt="Preview" 
                  className="h-full w-full object-contain p-4" 
                />
              ) : (
                <>
                  <Upload className="w-10 h-10 text-stone-400 mb-4" />
                  <span className="text-stone-500 font-medium">Select Image</span>
                </>
              )}
            </div>
          </div>

          {error && (
            <div className="mt-4 flex items-center gap-2 text-red-600 text-sm">
              <AlertCircle className="w-4 h-4" />
              {error}
            </div>
          )}

          <button 
            onClick={handleAnalyze} 
            disabled={!file || loading || error}
            className={`
              mt-8 w-full py-4 px-6 rounded-lg font-bold flex items-center justify-center gap-2 transition-all uppercase tracking-widest text-sm
              ${!file || loading || error
                ? 'bg-stone-200 text-stone-400 cursor-not-allowed' 
                : 'bg-emerald-900 text-white hover:bg-emerald-800 shadow-md'}
            `}
          >
            {loading ? (
              <>
                <Loader2 className="w-4 h-4 animate-spin" /> Processing...
              </>
            ) : (
              <>
                <Scan className="w-4 h-4" /> Identify Item
              </>
            )}
          </button>
        </div>

        {/* Right Side - Results Area */}
        <div className="w-full md:w-1/2 p-8 md:p-12 bg-white flex flex-col justify-center border-l border-stone-100">
          {!result ? (
            <div className="text-center text-stone-300 flex flex-col items-center">
              <Gem className="w-16 h-16 mb-4 opacity-20" />
              <p className="font-serif italic">Analysis details will appear here</p>
            </div>
          ) : (
            <div className="animate-fade-in">
              {/* <div className="inline-block px-3 py-1 bg-emerald-100 text-emerald-800 text-xs font-bold mb-6 uppercase tracking-widest rounded-sm">
                Verified Result
              </div> */}
              
              <div className="mb-10">
                <h3 className="text-xs font-bold text-stone-400 uppercase tracking-widest mb-2">Type</h3>
                <p className="text-4xl font-serif text-stone-900 border-b-2 border-emerald-500 inline-block pb-2">
                  {result.type}
                </p>
              </div>
              
              <div>
                <h3 className="text-xs font-bold text-stone-400 uppercase tracking-widest mb-3">Description</h3>
                <p className="text-stone-600 leading-loose font-light text-lg">
                  {result.description}
                </p>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

export default App;