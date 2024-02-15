using System.Collections;
using UnityEngine;
using UnityEngine.UI;
using Mediapipe.Unity.CoordinateSystem;

using Stopwatch = System.Diagnostics.Stopwatch;
using TMPro;
using Unity.VisualScripting;
using UnityEngine.UIElements;

namespace Mediapipe.Unity.EyeTrackingSystem
{
    public class FaceLandmarksParser : MonoBehaviour
    {
        [SerializeField] private TextAsset _configAsset;
        [SerializeField] private RawImage _screen;
        [SerializeField] private int _width;
        [SerializeField] private int _height;
        [SerializeField] private int _fps;

        private CalculatorGraph _graph;
        private OutputStream<ImageFrame> _outputVideoStream;
        private OutputStream<NormalizedLandmarkList> _FaceandIrisLandmarksStream;
        private ResourceManager _resourceManager;

        private WebCamTexture _webcamTexture;

        private Texture2D _inputTexture;
        private Color32[] _inputPixelData;
        private Texture2D _outputTexture;
        private Color32[] _outputPixelData;

        private IEnumerator Start()
        {
            if(WebCamTexture.devices.Length == 0)
            {
                throw new System.Exception("Web Camera Devices are not found");
            }

            var webcamDevice = WebCamTexture.devices[3];
            _webcamTexture = new WebCamTexture(webcamDevice.name, _width, _height, _fps);
            _webcamTexture.Play();

            yield return new WaitUntil(() => _webcamTexture.width > 16);

            _screen.rectTransform.sizeDelta = new Vector2(_width, _height);

            _inputTexture = new Texture2D(_width, _height, TextureFormat.RGBA32, false);
            _inputPixelData = new Color32[_width*_height];
            _outputTexture = new Texture2D(_width, _height,TextureFormat.RGBA32, false);
            _outputPixelData = new Color32[_width*_height];

            _screen.texture = _outputTexture;

            _resourceManager = new LocalResourceManager();
            yield return _resourceManager.PrepareAssetAsync("face_detection_short_range.bytes");
            yield return _resourceManager.PrepareAssetAsync("face_landmark.bytes");
            yield return _resourceManager.PrepareAssetAsync("iris_landmark.bytes");

            var stopwatch = new Stopwatch();
            
            Protobuf.SetLogHandler(Protobuf.DefaultLogHandler);
            _graph = new CalculatorGraph(_configAsset.text);
            _outputVideoStream = new OutputStream<ImageFrame>(_graph, "output_video");
            _FaceandIrisLandmarksStream = new OutputStream<NormalizedLandmarkList>(_graph, "face_landmarks_with_iris");
            _outputVideoStream.StartPolling();
            _FaceandIrisLandmarksStream.StartPolling();
            _graph.StartRun();
            stopwatch.Start();

            var screenRect = _screen.GetComponent<RectTransform>().rect;

            while (true){
                _inputTexture.SetPixels32(_webcamTexture.GetPixels32(_inputPixelData));
                var imageFrame = new ImageFrame(ImageFormat.Types.Format.Srgba, _width, _height, _width * 4, _inputTexture.GetRawTextureData<byte>());
                var currentTimeStamp = stopwatch.ElapsedTicks / (System.TimeSpan.TicksPerMillisecond / 1000);
                _graph.AddPacketToInputStream("input_video", Packet.CreateImageFrameAt(imageFrame, currentTimeStamp));

                var task1 = _outputVideoStream.WaitNextAsync();
                var task2 = _FaceandIrisLandmarksStream.WaitNextAsync();
                yield return new WaitUntil(() => task1.IsCompleted && task2.IsCompleted);
                var result1 = task1.Result;
                var result2 = task2.Result;

                if (!result1.ok || !result2.ok)
                {
                    throw new System.Exception("Something Went Wrong");
                }

                var outputVideoPacket = result1.packet;
                if(outputVideoPacket != null )
                {
                    var outputVideo = outputVideoPacket.Get();

                    if (outputVideo.TryReadPixelData(_outputPixelData))
                    {
                        _outputTexture.SetPixels32(_outputPixelData);
                        _outputTexture.Apply();
                    }
                }

                // 얼굴 랜드마크 정보 출력은 여기서
                var FaceandIrisLandmarksPacket = result2.packet;
                if(FaceandIrisLandmarksPacket != null)
                {
                    var FaceandIrisLandmarks = FaceandIrisLandmarksPacket.Get(NormalizedLandmarkList.Parser);
                    if(FaceandIrisLandmarks != null)
                    {
                        //var topOfHead = FaceandIrisLandmarks.Landmark[10];
                        //Debug.Log($"Unity Local Coordinates: {screenRect.GetPoint(topOfHead)}, Image Coordinates: {topOfHead}");

                        // 안면 기울기 추적 함수
                        GetComponent<GazeEstimator>().HeadPoseTracker(FaceandIrisLandmarks);

                        //// 랜드마크 점을 구체가 따라다니게 함
                        //GetComponent<GazeEstimator>().FollowSphereToLandmarks(FaceandIrisLandmarks);

                        // 눈동자 추적 함수
                        GetComponent<GazeEstimator>().EyeTracker(FaceandIrisLandmarks);
                    }
                }

                yield return new WaitForEndOfFrame();
            }
        }

        private void OnDestroy()
        {
            if(_webcamTexture != null)
            {
                _webcamTexture.Stop();
            }

            _outputVideoStream?.Dispose();
            _outputPixelData = null;

            if(_graph != null)
            {
                try
                {
                    _graph.CloseInputStream("input_video");
                    _graph.WaitUntilDone();
                }
                finally
                {
                    _graph.Dispose();
                    _graph = null;
                }
            }
        }

        private void OnApplicationQuit()
        {
            Protobuf.ResetLogHandler();
        }
    }
}