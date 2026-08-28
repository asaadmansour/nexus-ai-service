FROM postgres:15-alpine AS build

ARG PGVECTOR_VERSION=0.8.6
ARG PGVECTOR_SHA256=10bf9938906e5d643bbc4a7eea104b6f57ba4898e5b76b20e60484ea1d5a7f8f

# The official Postgres Alpine image includes matching server headers. Build the
# extension against that exact server, then remove every compiler/build package.
RUN apk add --no-cache su-exec \
  && sed -i 's/exec gosu /exec su-exec /' /usr/local/bin/docker-entrypoint.sh \
  && rm -f /usr/local/bin/gosu \
  && apk add --no-cache --virtual .pgvector-build-deps \
      build-base ca-certificates curl \
  && curl --fail --location --retry 3 \
      "https://github.com/pgvector/pgvector/archive/refs/tags/v${PGVECTOR_VERSION}.tar.gz" \
      --output /tmp/pgvector.tar.gz \
  && echo "${PGVECTOR_SHA256}  /tmp/pgvector.tar.gz" | sha256sum -c - \
  && mkdir /tmp/pgvector \
  && tar --extract --gzip --file /tmp/pgvector.tar.gz \
      --directory /tmp/pgvector --strip-components=1 \
  && make --directory /tmp/pgvector OPTFLAGS="" with_llvm=no \
  && make --directory /tmp/pgvector install with_llvm=no \
  && rm -rf /tmp/pgvector /tmp/pgvector.tar.gz \
  && apk del .pgvector-build-deps

# Flatten the runtime filesystem so deleted build tools and the replaced Go
# helper do not remain in lower image layers or their vulnerability metadata.
FROM scratch
COPY --from=build / /

ENV LANG=en_US.utf8 \
    PG_MAJOR=15 \
    PGDATA=/var/lib/postgresql/data
EXPOSE 5432
VOLUME ["/var/lib/postgresql/data"]
ENTRYPOINT ["docker-entrypoint.sh"]
CMD ["postgres"]
STOPSIGNAL SIGINT
