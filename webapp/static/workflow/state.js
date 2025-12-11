/**
 * Workflow State Management
 * 
 * Centralized state management for the workflow application.
 * All modules should access state through this module.
 */

const WorkflowState = {
  // Workflow state
  currentFrameDir: null,
  currentSessionId: null,
  currentTrainingJobId: null,
  currentUploadId: null,
  currentVideoName: null,

  // Video metadata
  holdColor: '',
  routeDifficulty: '',
  firstFrameImageUrl: null,
  videoPreviewUrl: null, // URL for video preview element
  frameAspectRatio: null, // 'vertical' or 'horizontal'

  // Video trim state
  videoTrimStart: null,
  videoTrimEnd: null,
  videoDuration: null,
  originalFirstFrameUrl: null, // First frame of original video
  trimmedFirstFrameUrl: null,  // First frame at trim start position
  selectedFrameForSegmentation: null, // 'original' or 'trimmed'

  // 3D viewer (currently disabled)
  threeViewer: null,
  current3DModelId: null,

  // Frame selection state
  frameSelectionState: {
    uploadId: null,
    videoName: null,
    frames: [],
    currentIndex: 0,
    selectedFrames: new Set(),
    selectedFramesOrder: [], // Track selection order (frame indices)
    viewMode: 'all', // 'all' or 'selected'
    autoSelectedFirstFrame: false, // Track if first frame was auto-selected
  },

  // Hold labeling state
  holdLabelingSegments: [],
  holdLabelingCanvas: null,
  holdLabelingCtx: null,
  holdLabelingImage: null,
  selectedSegmentId: null,
  selectedSegmentIds: new Set(),
  holdLabelsSubmitted: false,

  // Training pool state
  frameSelectionSavedToPool: false,

  // Getters
  getCurrentFrameDir() {
    return this.currentFrameDir;
  },

  getCurrentSessionId() {
    return this.currentSessionId;
  },

  getCurrentTrainingJobId() {
    return this.currentTrainingJobId;
  },

  getCurrentUploadId() {
    return this.currentUploadId;
  },

  getCurrentVideoName() {
    return this.currentVideoName;
  },

  getFrameSelectionState() {
    return this.frameSelectionState;
  },

  getFrameAspectRatio() {
    return this.frameAspectRatio;
  },

  getHoldLabelingSegments() {
    return this.holdLabelingSegments;
  },

  // Setters
  setCurrentFrameDir(value) {
    this.currentFrameDir = value;
  },

  setCurrentSessionId(value) {
    this.currentSessionId = value;
  },

  setCurrentTrainingJobId(value) {
    this.currentTrainingJobId = value;
  },

  setCurrentUploadId(value) {
    this.currentUploadId = value;
  },

  setCurrentVideoName(value) {
    this.currentVideoName = value;
  },

  setHoldColor(value) {
    this.holdColor = value;
  },

  setRouteDifficulty(value) {
    this.routeDifficulty = value;
  },

  setFirstFrameImageUrl(value) {
    this.firstFrameImageUrl = value;
  },

  setVideoPreviewUrl(value) {
    this.videoPreviewUrl = value;
  },

  setFrameAspectRatio(value) {
    this.frameAspectRatio = value;
  },

  getFrameAspectRatio() {
    return this.frameAspectRatio;
  },

  // Video trim getters/setters
  setVideoTrimStart(value) {
    this.videoTrimStart = value;
  },

  getVideoTrimStart() {
    return this.videoTrimStart;
  },

  setVideoTrimEnd(value) {
    this.videoTrimEnd = value;
  },

  getVideoTrimEnd() {
    return this.videoTrimEnd;
  },

  setVideoDuration(value) {
    this.videoDuration = value;
  },

  getVideoDuration() {
    return this.videoDuration;
  },

  setOriginalFirstFrameUrl(value) {
    this.originalFirstFrameUrl = value;
  },

  getOriginalFirstFrameUrl() {
    return this.originalFirstFrameUrl;
  },

  setTrimmedFirstFrameUrl(value) {
    this.trimmedFirstFrameUrl = value;
  },

  getTrimmedFirstFrameUrl() {
    return this.trimmedFirstFrameUrl;
  },

  setSelectedFrameForSegmentation(value) {
    this.selectedFrameForSegmentation = value;
  },

  getSelectedFrameForSegmentation() {
    return this.selectedFrameForSegmentation;
  },

  // Reset methods
  resetFrameSelection() {
    this.frameSelectionState = {
      uploadId: null,
      videoName: null,
      frames: [],
      currentIndex: 0,
      selectedFrames: new Set(),
      selectedFramesOrder: [],
      viewMode: 'all',
      autoSelectedFirstFrame: false,
    };
  },

  resetHoldLabeling() {
    this.holdLabelingSegments = [];
    this.holdLabelingCanvas = null;
    this.holdLabelingCtx = null;
    this.holdLabelingImage = null;
    this.selectedSegmentId = null;
    this.selectedSegmentIds = new Set();
    this.holdLabelsSubmitted = false;
  },

  reset() {
    this.currentFrameDir = null;
    this.currentSessionId = null;
    this.currentTrainingJobId = null;
    this.currentUploadId = null;
    this.currentVideoName = null;
    this.holdColor = '';
    this.routeDifficulty = '';
    this.firstFrameImageUrl = null;
    this.videoPreviewUrl = null;
    this.frameAspectRatio = null;
    this.frameSelectionSavedToPool = false;
    // Reset trim state
    this.videoTrimStart = null;
    this.videoTrimEnd = null;
    this.videoDuration = null;
    this.originalFirstFrameUrl = null;
    this.trimmedFirstFrameUrl = null;
    this.selectedFrameForSegmentation = null;
    this.resetFrameSelection();
    this.resetHoldLabeling();
  },

  getHoldLabelsSubmitted() {
    return this.holdLabelsSubmitted;
  },

  setHoldLabelsSubmitted(value) {
    this.holdLabelsSubmitted = value;
  }
};
