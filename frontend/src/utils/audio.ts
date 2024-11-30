export const getAudioBlob = async (url: string): Promise<string> => {
    const token = localStorage.getItem('token');
    const response = await fetch(url, {
      headers: {
        'Authorization': `Bearer ${token}`
      }
    });
    
    if (!response.ok) {
      throw new Error('Failed to fetch audio');
    }
    
    const blob = await response.blob();
    return URL.createObjectURL(blob);
  };