FROM ubuntu:24.04 AS build

ARG BUILD_THREAD=64

RUN apt update && \
    apt install -y \
        build-essential libboost-all-dev libsqlite3-dev libasio-dev nasm cmake \
        ffmpeg libswscale-dev libavutil-dev libavcodec-dev libavdevice-dev libavfilter-dev libavformat-dev \
        libavutil-dev libpostproc-dev libswresample-dev \
        libhdf5-dev libc6-dev

WORKDIR /app

COPY . .

RUN cd x264; make -j $BUILD_THREAD; cd ..
RUN cd zstd; make -j $BUILD_THREAD; cd ..
RUN cd ffmpeg_HDF5_filter; cmake .; make -j $BUILD_THREAD; cd ..

# SANITIZE=ON builds the server with ASan + UBSan (see CMakeLists.txt).
ARG SANITIZE=OFF
RUN cmake -DNTRACER_SANITIZE=$SANITIZE .; exit 0
RUN make -j $BUILD_THREAD

# Runtime image: only the binary, the two in-tree shared libraries it loads
# through its RUNPATH (/app/zstd/lib, /app/ffmpeg_HDF5_filter), and the distro
# runtime libraries. curl is here so a health check can make a real request.
FROM ubuntu:24.04

ARG CDN_PORT=6000

RUN apt update && \
    apt install -y --no-install-recommends \
        libsqlite3-0 libavcodec60 libavformat60 libavutil58 libswscale7 \
        libhdf5-103-1t64 curl ca-certificates && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY --from=build /app/nTracer_cdn /app/nTracer_cdn
COPY --from=build /app/zstd/lib/libzstd.so* /app/zstd/lib/
COPY --from=build /app/ffmpeg_HDF5_filter/libh5ffmpeg_shared.so /app/ffmpeg_HDF5_filter/

EXPOSE ${CDN_PORT}

# exec replaces the shell, so the server is PID 1 and receives SIGTERM from
# docker stop instead of being killed after the timeout.
CMD ["sh", "-c", "ls -lh /data/; exec ./nTracer_cdn 6000 /data/"]
