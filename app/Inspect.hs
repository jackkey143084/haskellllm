-- | CLI: inspect a safetensors shard — print config summary and tensor list.
-- Usage: hq-inspect <model.safetensors> [config.json]
module Main (main) where

import qualified Data.ByteString as BS
import qualified Data.ByteString.Lazy as LBS
import Data.List (intercalate, sortBy)
import Data.Ord (comparing)
import System.Environment (getArgs)
import System.Exit (exitFailure)
import Control.Monad (forM_)
import Text.Printf (printf)
import HQ.JSON
import HQ.SafeTensors

main :: IO ()
main = do
  args <- getArgs
  case args of
    (shardPath : cfgRest) -> do
      r <- open shardPath
      case r of
        Left e -> die e
        Right st -> do
          let tis = sortBy (comparing negateBytes) [ t | (_, Just t) <- map (\n -> (n, tensorInfo st n)) (tensorNames st) ]
              negateBytes t = negate (tiLen t)
          printf "tensors: %d\n" (length tis)
          case cfgRest of
            (cfgPath : _) -> do
              cfgRaw <- LBS.readFile cfgPath
              either die showConfig (parseJSON cfgRaw)
            [] -> pure ()
          putStrLn "top 20 tensors by bytes:"
          forM_ (take 20 tis) $ \t ->
            printf "  %-60s %-5s %-30s %10.1f KB\n"
              (tiName t) (show (tiDtype t)) (show (tiShape t)) (fromIntegral (tiLen t) / 1024 :: Double)
        _ -> die "unreachable"
    _ -> do
      putStrLn "usage: hq-inspect <model.safetensors> [config.json]"
      exitFailure

showConfig :: JValue -> IO ()
showConfig v = do
  -- Qwen3.5 nests the LLM fields under text_config
  let tc = maybe v id (lookupKey "text_config" v)
      get k = lookupKey k tc >>= jToInt
  putStrLn "config:"
  forM_ ["hidden_size","num_hidden_layers","num_attention_heads",
         "num_key_value_heads","intermediate_size","vocab_size"] $ \k ->
    printf "  %-22s %s\n" k (maybe "?" show (get k))

die :: String -> IO a
die e = putStrLn ("error: " ++ e) >> exitFailure
