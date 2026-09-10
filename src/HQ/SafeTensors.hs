-- | A safetensors file reader. Zero dependencies beyond GHC boot libraries.
--
-- Format: 8-byte little-endian u64 header length, then a JSON header of
-- { "tensor.name": {"dtype": "...", "shape": [...], "data_offsets": [s, e]},
--   "__metadata__": {...} }, then the raw tensor data region.
--
-- Reads are pure file seeks (the handle is held open), so tensors are
-- effectively mmapped: no full-file copy, which is load-bearing on the
-- 3 GB cgroup box (heap copies OOM at 2x peak — measured in the Go port).
module HQ.SafeTensors
  ( Dtype(..)
  , TensorInfo(..)
  , SafeTensorsFile
  , open
  , tensorNames
  , tensorInfo
  , tensorBytes
  , readAsFloat
  , le64
  , unpack16
  , unpack32
  , word32ToFloat
  , halfToFloat
  , safeSlice
  ) where

import qualified Data.ByteString as BS
import qualified Data.ByteString.Lazy as LBS
import Data.Bits (shiftL, shiftR, (.&.), (.|.))
import Data.Word (Word16, Word64)
import System.IO
import Control.Monad (forM)
import GHC.Float (castWord32ToFloat)
import HQ.BF16 (BF16(..), bf16ToF32)
import HQ.JSON

data Dtype = DTBF16 | DTF32 | DTF16 | DTOther String
  deriving (Eq, Show)

data TensorInfo = TensorInfo
  { tiName  :: !String
  , tiDtype :: !Dtype
  , tiShape :: ![Int]
  , tiStart :: !Word64   -- ^ absolute offset in file
  , tiLen   :: !Word64   -- ^ byte length
  } deriving (Eq, Show)

data SafeTensorsFile = SafeTensorsFile
  { stHandle  :: !Handle
  , stTensors :: [(String, TensorInfo)]
  }

-- | Open a safetensors file and parse its header.
open :: FilePath -> IO (Either String SafeTensorsFile)
open path = do
  h <- openBinaryFile path ReadMode
  hdrLenBS <- BS.hGet h 8
  if BS.length hdrLenBS < 8
    then do hClose h; pure (Left "file too short for safetensors header")
    else do
      let hdrLen = le64 hdrLenBS
      hdrBS <- LBS.hGet h (fromIntegral hdrLen)
      case parseJSON hdrBS of
        Left e -> do hClose h; pure (Left ("bad header JSON: " ++ e))
        Right (JObj kvs) -> do
          let tensors = [ (k, v) | (k, v) <- kvs, k /= "__metadata__" ]
          case mapM (\(k, v) -> fmap (k,) (parseTensor (8 + hdrLen) k v)) tensors of
            Left e -> do hClose h; pure (Left e)
            Right tis -> pure (Right (SafeTensorsFile h tis))
        Right _ -> do hClose h; pure (Left "header is not a JSON object")

-- | Decode a little-endian u64.
le64 :: BS.ByteString -> Word64
le64 bs =
  foldr (\i acc -> (acc `shiftL` 8) .|. fromIntegral (BS.index bs i)) 0 [0..7]

parseTensor :: Word64 -> String -> JValue -> Either String TensorInfo
parseTensor base name (JObj kvs) = do
  dt <- case lookup "dtype" kvs >>= jToString of
    Just s -> Right (dtypeFromString s)
    Nothing -> Left ("tensor " ++ name ++ ": missing dtype")
  shape <- case lookup "shape" kvs >>= jToArray of
    Just xs -> mapM (maybe (Left "bad shape") Right . jToInt) xs
    Nothing -> Left ("tensor " ++ name ++ ": missing shape")
  (off0, off1) <- case lookup "data_offsets" kvs >>= jToArray of
    Just [a, b] -> do
      a' <- maybe (Left "bad offset") Right (jToInt a)
      b' <- maybe (Left "bad offset") Right (jToInt b)
      Right (fromIntegral a' :: Word64, fromIntegral b' :: Word64)
    _ -> Left ("tensor " ++ name ++ ": bad data_offsets")
  pure TensorInfo
    { tiName = name
    , tiDtype = dt
    , tiShape = shape
    , tiStart = base + off0
    , tiLen = off1 - off0
    }
parseTensor _ name _ = Left ("tensor " ++ name ++ ": not an object")

dtypeFromString :: String -> Dtype
dtypeFromString "BF16" = DTBF16
dtypeFromString "F32"  = DTF32
dtypeFromString "F16"  = DTF16
dtypeFromString s      = DTOther s

-- | All tensor names in file order.
tensorNames :: SafeTensorsFile -> [String]
tensorNames = map fst . stTensors

-- | Info for one tensor.
tensorInfo :: SafeTensorsFile -> String -> Maybe TensorInfo
tensorInfo st n = lookup n (stTensors st)

-- | Raw bytes of a tensor (read from the open handle at its offset).
tensorBytes :: SafeTensorsFile -> String -> IO (Maybe BS.ByteString)
tensorBytes st name = case tensorInfo st name of
  Nothing -> pure Nothing
  Just ti -> do
    hSeek (stHandle st) AbsoluteSeek (fromIntegral (tiStart ti))
    Just <$> BS.hGet (stHandle st) (fromIntegral (tiLen ti))

-- | Read a tensor as [Float], decoding BF16/F16/F32, in file order.
readAsFloat :: SafeTensorsFile -> String -> IO (Maybe [Float])
readAsFloat st name = case tensorInfo st name of
  Nothing -> pure Nothing
  Just ti -> do
    mb <- tensorBytes st name
    pure (fmap (decode ti) mb)
  where
    decode ti bs = case tiDtype ti of
      DTBF16 -> map (bf16ToF32 . BF16) (unpack16 bs)
      DTF32  -> map word32ToFloat (unpack32 bs)
      DTF16  -> map (halfToFloat . fromIntegral) (unpack16 bs)
      DTOther _ -> []

-- | Unpack little-endian 16-bit words, in file order.
unpack16 :: BS.ByteString -> [Word16]
unpack16 bs = reverse (go 0 [])
  where
    n = BS.length bs
    go :: Int -> [Word16] -> [Word16]
    go i acc
      | i + 2 > n = acc
      | otherwise = go (i + 2)
          (( fromIntegral (BS.index bs i)
             .|. (fromIntegral (BS.index bs (i+1)) `shiftL` 8)) : acc)

-- | Unpack little-endian 32-bit words, in file order.
unpack32 :: BS.ByteString -> [Word64]
unpack32 bs = reverse (go 0 [])
  where
    n = BS.length bs
    go i acc
      | i + 4 > n = acc
      | otherwise = go (i + 4)
          (( fromIntegral (BS.index bs i)
             .|. (fromIntegral (BS.index bs (i+1)) `shiftL` 8)
             .|. (fromIntegral (BS.index bs (i+2)) `shiftL` 16)
             .|. (fromIntegral (BS.index bs (i+3)) `shiftL` 24)) : acc)

-- | Reinterpret a 32-bit pattern as an IEEE-754 single-precision float.
word32ToFloat :: Word64 -> Float
word32ToFloat = castWord32ToFloat . fromIntegral

-- | Half-precision (IEEE F16) to float, exact.
halfToFloat :: Word64 -> Float
halfToFloat h =
  let sign  = (h `shiftR` 15) .&. 1
      exp10 = (h `shiftR` 10) .&. 0x1f
      man10 = h .&. 0x3ff
  in if exp10 == 0
       then subnormal sign man10
       else if exp10 == 31
         then special sign man10
         else
           let e = fromIntegral exp10 - 15 + 127 :: Word64
               bits = (sign `shiftL` 31) .|. (e `shiftL` 23) .|. (man10 `shiftL` 13)
           in word32ToFloat bits
  where
    signF :: Word64 -> Float
    signF s = if s == 1 then -1.0 else 1.0
    subnormal :: Word64 -> Word64 -> Float
    subnormal sign man = signF sign * fromIntegral man * 6.103515625e-8  -- 2^-24
    special :: Word64 -> Word64 -> Float
    special sign man
      | man == 0  = signF sign * (1.0/0.0)
      | otherwise = 0.0/0.0

-- | Bounds-checked slice helper.
safeSlice :: BS.ByteString -> Int -> Int -> Maybe BS.ByteString
safeSlice bs off n
  | off < 0 || n < 0 || off + n > BS.length bs = Nothing
  | otherwise = Just (BS.take n (BS.drop off bs))
