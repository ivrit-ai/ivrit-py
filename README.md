# ivrit-py

## RunPod Payload Size Limit

When using RunPod, be aware that there is a payload size limit for API requests. If your payload exceeds this limit, the request will fail. Here are some workarounds:

### Workarounds

1. **Chunking**: Split large payloads into smaller chunks and process them sequentially or in parallel using multiple requests.
2. **Cloud Storage**: Upload large payloads to cloud storage (e.g., S3, GCS) and pass the download URL to RunPod instead of the raw data.
3. **Compression**: Compress data before transmission to reduce payload size (e.g., using gzip or zlib).
4. **Streaming**: For very large files, use streaming APIs if available to avoid loading the entire payload into memory.

### Error Handling

If you encounter a payload size error, check the RunPod documentation for current limits and implement one of the above strategies. The error message typically includes the maximum allowed size.

For more details, refer to the [RunPod API documentation](https://docs.runpod.com/).
<<<ENDFILE>>
