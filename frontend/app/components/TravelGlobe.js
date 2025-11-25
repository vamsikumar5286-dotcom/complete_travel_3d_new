"use client";

import dynamic from "next/dynamic";
import { Suspense } from "react";

const Globe = dynamic(() => import("react-globe.gl"), { ssr: false });

export default function TravelGlobe() {
  return (
    <div style={{ width: "100%", height: "100vh" }}>
      <Suspense fallback={<div>Loading globe...</div>}>
        <Globe
          globeImageUrl="//unpkg.com/three-globe/example/img/earth-blue-marble.jpg"
          backgroundColor="#000011"
        />
      </Suspense>
    </div>
  );
}