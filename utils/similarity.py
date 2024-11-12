import cv2
     
def calculate_similarity(img1, img2):
    """
    Calculate the similarity between two images using ORB feature matching.
    This method is invariant to translation and rotation.
    
    :param img1: First image
    :param img2: Second image
    :return: A similarity score between 0 and 1
    """
    # Initialize ORB detector
    orb = cv2.ORB_create()
    
    # Find the keypoints and descriptors with ORB
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)
    
    # Create BFMatcher object
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    
    # Match descriptors
    matches = bf.match(des1, des2)
    
    # Sort them in the order of their distance
    matches = sorted(matches, key=lambda x: x.distance)
    
    # Calculate similarity score
    similarity = len(matches) / max(len(kp1), len(kp2))
    
    return similarity