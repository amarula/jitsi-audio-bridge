# Debian package build. See README.md, "Debian package".
#
# Build on the machine (or a container matching it) where the package will be
# installed: the bundled virtualenv belongs to one python3 minor version and
# architecture.
.PHONY: deb
deb:
	packaging/build-deb.sh
