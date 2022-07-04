import { NgModule } from '@angular/core';
import { BrowserModule } from '@angular/platform-browser';

import { AppComponent } from './app.component';
import { HttpClientModule } from '@angular/common/http';
import { UploadWindowComponent } from './upload-window/upload-window.component';
import { ImageResultComponent } from './image-result/image-result.component';

@NgModule({
  declarations: [
    AppComponent,
    UploadWindowComponent,
    ImageResultComponent
  ],
  imports: [
    BrowserModule,
    HttpClientModule
  ],
  providers: [],
  bootstrap: [AppComponent]
})
export class AppModule { }
