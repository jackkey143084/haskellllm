-- | Self-tests for the HQ substrate modules (BF16, JSON, SafeTensors).
-- Run: make test — exits 0 on green.
--
-- Expected bf16 values cross-checked against Python:
--   struct.unpack('>f', struct.pack('>I', bits << 16))[0]
module Main (main) where

import qualified Data.ByteString.Lazy.Char8 as LC
import Data.Bits (shiftL, shiftR, (.&.), (.|.))
import Data.IORef
import Data.Word (Word16, Word32, Word64)
import System.Directory (removeFile)
import Control.Monad (forM_, unless)
import GHC.Float (castWord32ToFloat, castFloatToWord32)
import HQ.BF16
import HQ.JSON
import HQ.SafeTensors

main :: IO ()
main = do
  ref <- newIORef (0 :: Int, 0 :: Int)
  let check name cond = do
        modifyIORef ref (\(p, f) -> if cond then (p+1, f) else (p, f+1))
        unless cond (putStrLn ("FAIL: " ++ name))
  testBF16 check
  testJSON check
  testSafetensors check
  (p, f) <- readIORef ref
  putStrLn ("passed " ++ show p ++ ", failed " ++ show f)
  unless (f == 0) (ioError (userError "TESTS FAILED"))

testBF16 :: (String -> Bool -> IO ()) -> IO ()
testBF16 check = do
  -- only bf16-representable values roundtrip exactly
  let reps = [0.0, -0.0, 1.0, -1.0, 0.5, 2.0, 1.5, -3.25, 1.0078125, 65536.0, 6.103515625e-5]
  forM_ reps $ \x ->
    check ("bf16 roundtrip " ++ show x) (bf16ToF32 (f32ToBF16 x) == x)
  -- known bit patterns (bf16 is the top half of f32)
  check "1.0 bits"  (bf16ToF32 (BF16 0x3F80) == 1.0)
  check "-1.0 bits" (bf16ToF32 (BF16 0xBF80) == -1.0)
  check "0.5 bits"  (bf16ToF32 (BF16 0x3F00) == 0.5)
  check "2.0 bits"  (bf16ToF32 (BF16 0x4000) == 2.0)
  check "encode 1.0"  (f32ToBF16 1.0 == BF16 0x3F80)
  check "encode -1.0" (f32ToBF16 (-1.0) == BF16 0xBF80)
  -- 1.00390625 = 1 + 2^-8 sits below the halfway point to 1 + 2^-7, rounds down to 1.0
  check "rne down" (f32ToBF16 1.00390625 == BF16 0x3F80)
  -- exact tie halfway between 0x3F80 and 0x3F81 rounds to even (0x3F80)
  let tieF = castWord32ToFloat ((0x3F80 `shiftL` 16) + 0x8000)
  check "rne tie-to-even" (f32ToBF16 tieF == BF16 0x3F80)
  -- odd-tie case: halfway between 0x3F81 and 0x3F82 -> even 0x3F82
  let tieF2 = castWord32ToFloat ((0x3F81 `shiftL` 16) + 0x8000)
  check "rne tie-to-even-odd" (f32ToBF16 tieF2 == BF16 0x3F82)
  check "inf roundtrip" (bf16ToF32 (f32ToBF16 (1.0/0.0)) == 1.0/0.0)

testJSON :: (String -> Bool -> IO ()) -> IO ()
testJSON check = do
  check "json null" (parseJSON (LC.pack "null") == Right JNull)
  check "json true" (parseJSON (LC.pack "true") == Right (JBool True))
  check "json num" (parseJSON (LC.pack "3.5") == Right (JNum 3.5))
  check "json neg int" (parseJSON (LC.pack "-42") == Right (JNum (-42)))
  check "json str" (parseJSON (LC.pack "\"hi\"") == Right (JStr "hi"))
  check "json str escapes" (parseJSON (LC.pack "\"a\\nb\\u0041\"") == Right (JStr "a\nbA"))
  check "json arr" (parseJSON (LC.pack "[1, 2, 3.5]") == Right (JArr [JNum 1, JNum 2, JNum 3.5]))
  check "json obj" (parseJSON (LC.pack "{\"a\": 1, \"b\": [true]}")
                      == Right (JObj [("a", JNum 1), ("b", JArr [JBool True])]))
  check "json ws" (parseJSON (LC.pack "  { \"x\" : [ 1 ] }  ")
                      == Right (JObj [("x", JArr [JNum 1])]))
  check "json trailing garbage rejected"
    (case parseJSON (LC.pack "1 2") of Left _ -> True; Right _ -> False)
  check "json lookupKey"
    (lookupKey "shape" (JObj [("shape", JArr [JNum 2, JNum 8])]) == Just (JArr [JNum 2, JNum 8]))

-- | Build a synthetic safetensors file with one BF16 and one F32 tensor,
-- read it back, compare decoded values.
testSafetensors :: (String -> Bool -> IO ()) -> IO ()
testSafetensors check = do
  let path = "/tmp/hq_selftest.safetensors"
      bfVals = [1.0, -1.0, 0.5, 2.0] :: [Float]
      f32Vals = [3.25, -0.125] :: [Float]
      bfData = concatMap (le16 . bits16 . f32ToBF16) bfVals
      f32Data = concatMap le32 f32Vals
      hdr = "{\"w.bf16\": {\"dtype\": \"BF16\", \"shape\": [4], \"data_offsets\": [0, 8]}"
            ++ ",\"w.f32\": {\"dtype\": \"F32\", \"shape\": [2], \"data_offsets\": [8, 16]}}"
      hdrLen = length hdr
      file = LC.pack (le64str (fromIntegral hdrLen) ++ hdr) <> LC.pack (bfData ++ f32Data)
  LC.writeFile path file
  r <- open path
  case r of
    Left e -> check ("safetensors open: " ++ e) False
    Right st -> do
      check "st names" (tensorNames st == ["w.bf16", "w.f32"])
      bf <- readAsFloat st "w.bf16"
      check "st bf16 values" (bf == Just bfVals)
      f32 <- readAsFloat st "w.f32"
      check "st f32 values" (f32 == Just f32Vals)
      check "st shape" (fmap tiShape (tensorInfo st "w.bf16") == Just [4])
  removeFile path

-- little-endian encoders for the synthetic file
le16 :: Word16 -> String
le16 w = [toEnum (fromIntegral (w .&. 0xff)), toEnum (fromIntegral (w `shiftR` 8))]

le32 :: Float -> String
le32 f =
  let w = fromIntegral (castFloatToWord32 f) :: Word64
  in [ toEnum (fromIntegral (w .&. 0xff))
     , toEnum (fromIntegral ((w `shiftR` 8) .&. 0xff))
     , toEnum (fromIntegral ((w `shiftR` 16) .&. 0xff))
     , toEnum (fromIntegral ((w `shiftR` 24) .&. 0xff)) ]

le64str :: Word64 -> String
le64str w = concat [le16 (fromIntegral (w `shiftR` (16*k)) .&. 0xffff) | k <- [0..3]]

bits16 :: BF16 -> Word16
bits16 (BF16 w) = w

bits32 :: Float -> Word64
bits32 = fromIntegral . castFloatToWord32
